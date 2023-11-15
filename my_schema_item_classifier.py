import logging
import os
import torch

from transformers import RobertaTokenizerFast, XLMRobertaTokenizerFast
from utils.classifier_model import MyClassifier
from transformers.trainer_utils import set_seed

_CLASSIFICATION_MODEL = None
_CLASSIFICATION_TOKENIZER = None


def get_classification_model():
    global _CLASSIFICATION_MODEL, _CLASSIFICATION_TOKENIZER
    if _CLASSIFICATION_MODEL is None:
        logging.info("Loading classification model")
        model_name_or_path = os.environ.get('T2S_CLASSIFICATION_MODEL')

        tokenizer_class = XLMRobertaTokenizerFast if "xlm" in model_name_or_path else RobertaTokenizerFast

        # load tokenizer
        _CLASSIFICATION_TOKENIZER = tokenizer_class.from_pretrained(
            model_name_or_path,
            add_prefix_space=True
        )

        # initialize model
        _CLASSIFICATION_MODEL = MyClassifier(
            model_name_or_path=model_name_or_path,
            vocab_size=len(_CLASSIFICATION_TOKENIZER),
            mode='test'
        )

        # load fine-tuned params
        _CLASSIFICATION_MODEL.load_state_dict(torch.load(model_name_or_path + "/dense_classifier.pt",
                                                         map_location=torch.device('cpu')), strict=False)
        if torch.cuda.is_available():
            _CLASSIFICATION_MODEL = _CLASSIFICATION_MODEL.cuda()
        _CLASSIFICATION_MODEL.eval()
        logging.info("Loaded classification model")
    return _CLASSIFICATION_TOKENIZER, _CLASSIFICATION_MODEL


def prepare_batch_inputs_and_labels(batch, tokenizer):
    batch_size = len(batch)

    batch_questions = [data[0] for data in batch]

    batch_table_names = [data[1] for data in batch]
    batch_table_labels = [data[2] for data in batch]

    batch_column_infos = [data[3] for data in batch]
    batch_column_labels = [data[4] for data in batch]

    batch_input_tokens, batch_column_info_ids, batch_table_name_ids, batch_column_number_in_each_table = [], [], [], []
    for batch_id in range(batch_size):
        input_tokens = [batch_questions[batch_id]]
        table_names_in_one_db = batch_table_names[batch_id]
        column_infos_in_one_db = batch_column_infos[batch_id]

        batch_column_number_in_each_table.append(
            [len(column_infos_in_one_table) for column_infos_in_one_table in column_infos_in_one_db])

        column_info_ids, table_name_ids = [], []

        for table_id, table_name in enumerate(table_names_in_one_db):
            input_tokens.append("|")
            input_tokens.append(table_name)
            table_name_ids.append(len(input_tokens) - 1)
            input_tokens.append(":")

            for column_info in column_infos_in_one_db[table_id]:
                input_tokens.append(column_info)
                column_info_ids.append(len(input_tokens) - 1)
                input_tokens.append(",")

            input_tokens = input_tokens[:-1]

        batch_input_tokens.append(input_tokens)
        batch_column_info_ids.append(column_info_ids)
        batch_table_name_ids.append(table_name_ids)

    # notice: the trunction operation will discard some tables and columns that exceed the max length
    tokenized_inputs = tokenizer(
        batch_input_tokens,
        return_tensors="pt",
        is_split_into_words=True,
        padding="max_length",
        max_length=512,
        truncation=True
    )

    batch_aligned_question_ids, batch_aligned_column_info_ids, batch_aligned_table_name_ids = [], [], []
    batch_aligned_table_labels, batch_aligned_column_labels = [], []

    # align batch_question_ids, batch_column_info_ids, and batch_table_name_ids after tokenizing
    for batch_id in range(batch_size):
        word_ids = tokenized_inputs.word_ids(batch_index=batch_id)

        aligned_question_ids, aligned_table_name_ids, aligned_column_info_ids = [], [], []
        aligned_table_labels, aligned_column_labels = [], []

        # align question tokens
        for token_id, word_id in enumerate(word_ids):
            if word_id == 0:
                aligned_question_ids.append(token_id)

        # align table names
        for t_id, table_name_id in enumerate(batch_table_name_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if table_name_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current table name
            if len(temp_list) != 0:
                aligned_table_name_ids.append(temp_list)
                aligned_table_labels.append(batch_table_labels[batch_id][t_id])

        # align column names
        for c_id, column_id in enumerate(batch_column_info_ids[batch_id]):
            temp_list = []
            for token_id, word_id in enumerate(word_ids):
                if column_id == word_id:
                    temp_list.append(token_id)
            # if the tokenizer doesn't discard current column name
            if len(temp_list) != 0:
                aligned_column_info_ids.append(temp_list)
                aligned_column_labels.append(batch_column_labels[batch_id][c_id])

        batch_aligned_question_ids.append(aligned_question_ids)
        batch_aligned_table_name_ids.append(aligned_table_name_ids)
        batch_aligned_column_info_ids.append(aligned_column_info_ids)
        batch_aligned_table_labels.append(aligned_table_labels)
        batch_aligned_column_labels.append(aligned_column_labels)

    # update column number in each table (because some tables and columns are discarded)
    for batch_id in range(batch_size):
        if len(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_table_labels[batch_id]):
            batch_column_number_in_each_table[batch_id] = batch_column_number_in_each_table[batch_id][
                                                          : len(batch_aligned_table_labels[batch_id])]

        if sum(batch_column_number_in_each_table[batch_id]) > len(batch_aligned_column_labels[batch_id]):
            truncated_column_number = sum(batch_column_number_in_each_table[batch_id]) - len(
                batch_aligned_column_labels[batch_id])
            batch_column_number_in_each_table[batch_id][-1] -= truncated_column_number

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    batch_aligned_column_labels = [torch.LongTensor(column_labels) for column_labels in batch_aligned_column_labels]
    batch_aligned_table_labels = [torch.LongTensor(table_labels) for table_labels in batch_aligned_table_labels]

    # print("\n".join(tokenizer.batch_decode(encoder_input_ids, skip_special_tokens = True)))

    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()
        batch_aligned_column_labels = [column_labels.cuda() for column_labels in batch_aligned_column_labels]
        batch_aligned_table_labels = [table_labels.cuda() for table_labels in batch_aligned_table_labels]

    return encoder_input_ids, encoder_input_attention_mask, \
        batch_aligned_column_labels, batch_aligned_table_labels, \
        batch_aligned_question_ids, batch_aligned_column_info_ids, \
        batch_aligned_table_name_ids, batch_column_number_in_each_table


def classification(batch, seed=42, device="0"):
    set_seed(seed)

    # tokenizer, model
    tokenizer, model = get_classification_model()

    table_labels_for_auc, column_labels_for_auc = [], []
    table_pred_probs_for_auc, column_pred_probs_for_auc = [], []

    returned_table_pred_probs, returned_column_pred_probs = [], []

    encoder_input_ids, encoder_input_attention_mask, \
        batch_column_labels, batch_table_labels, batch_aligned_question_ids, \
        batch_aligned_column_info_ids, batch_aligned_table_name_ids, \
        batch_column_number_in_each_table = prepare_batch_inputs_and_labels(batch, tokenizer)

    with torch.no_grad():
        model_outputs = model.forward(
            encoder_input_ids,
            encoder_input_attention_mask,
            batch_aligned_question_ids,
            batch_aligned_column_info_ids,
            batch_aligned_table_name_ids,
            batch_column_number_in_each_table
        )

    for batch_id, table_logits in enumerate(model_outputs["batch_table_name_cls_logits"]):
        table_pred_probs = torch.nn.functional.softmax(table_logits, dim = 1)
        returned_table_pred_probs.append(table_pred_probs[:, 1].cpu().tolist())

        table_pred_probs_for_auc.extend(table_pred_probs[:, 1].cpu().tolist())
        table_labels_for_auc.extend(batch_table_labels[batch_id].cpu().tolist())

    for batch_id, column_logits in enumerate(model_outputs["batch_column_info_cls_logits"]):
        column_number_in_each_table = batch_column_number_in_each_table[batch_id]
        column_pred_probs = torch.nn.functional.softmax(column_logits, dim = 1)
        returned_column_pred_probs.append([column_pred_probs[:, 1].cpu().tolist()[
                                           sum(column_number_in_each_table[:table_id]):sum(
                                               column_number_in_each_table[:table_id + 1])] \
                                           for table_id in range(len(column_number_in_each_table))])

        column_pred_probs_for_auc.extend(column_pred_probs[:, 1].cpu().tolist())
        column_labels_for_auc.extend(batch_column_labels[batch_id].cpu().tolist())

    return returned_table_pred_probs, returned_column_pred_probs


if __name__ == "__main__":
    os.environ['T2S_CLASSIFICATION_MODEL'] = './models/xlm_roberta_text2natsql_schema_item_classifier'
    test_batch = [(
        '我们有多少歌手？',
        ['stadium', 'singer', 'concert', 'singer in concert'],
        [0, 0, 0, 0],
        [
            ['stadium id', 'location', 'name', 'capacity', 'highest', 'lowest', 'average'],
            ['singer id', 'name', 'country', 'song name', 'song release year', 'age', 'is male'],
            ['concert id', 'concert name', 'theme', 'stadium id', 'year'],
            ['concert id', 'singer id']
        ],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    )]
    total_table_pred_probs, total_column_pred_probs = classification(batch=test_batch)
    print(total_table_pred_probs, total_column_pred_probs)
    total_table_pred_probs, total_column_pred_probs = classification(batch=test_batch)
    print(total_table_pred_probs, total_column_pred_probs)
