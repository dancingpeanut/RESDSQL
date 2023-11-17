import logging
import os

import torch
from tokenizers import AddedToken
from transformers import T5TokenizerFast, T5ForConditionalGeneration, MT5ForConditionalGeneration
from utils.text2sql_decoding_utils import decode_natsqls
from NatSQL.my_table_transform import transfer_table_info

_PREDICT_MODEL = None
_PREDICT_TOKENIZER = None


def get_resdsql_model():
    global _PREDICT_MODEL, _PREDICT_TOKENIZER
    if _PREDICT_MODEL is None:
        logging.info("Loading resdsql model")
        model_name_or_path = os.environ.get('T2S_RESDSQL_MODEL')

        # initialize tokenizer
        _PREDICT_TOKENIZER = T5TokenizerFast.from_pretrained(
            model_name_or_path,
            add_prefix_space=True
        )

        if isinstance(_PREDICT_TOKENIZER, T5TokenizerFast):
            _PREDICT_TOKENIZER.add_tokens([AddedToken(" <="), AddedToken(" <")])

        model_class = MT5ForConditionalGeneration if "mt5" in model_name_or_path else T5ForConditionalGeneration

        # initialize model
        _PREDICT_MODEL = model_class.from_pretrained(model_name_or_path)
        if torch.cuda.is_available():
            _PREDICT_MODEL = _PREDICT_MODEL.cuda()

        _PREDICT_MODEL.eval()
        logging.info("Loaded resdsql model")
    return _PREDICT_TOKENIZER, _PREDICT_MODEL


def get_db_path():
    return os.environ.get("T2S_DB_PATH")


def predict_with_natsql(question, nat_table):
    """
    只能有一个db，也就是
    """
    db_path = get_db_path()

    inputs = [question]
    tc_map = {}
    for f in nat_table['tc_fast']:
        tb_name = f.split('.')[0]
        if tb_name not in tc_map:
            tc_map[tb_name] = []
        tc_map[tb_name].append(f)
    for tb_name, tc in tc_map.items():
        inputs.append(f"{tb_name} : {' , '.join(tc)}")
    final_question = " | ".join(inputs)
    print("final_question", final_question)

    tokenizer, model = get_resdsql_model()
    tokenized_inputs = tokenizer(
        [final_question],
        return_tensors="pt",
        padding="max_length",
        max_length=512,
        truncation=True
    )

    encoder_input_ids = tokenized_inputs["input_ids"]
    encoder_input_attention_mask = tokenized_inputs["attention_mask"]
    if torch.cuda.is_available():
        encoder_input_ids = encoder_input_ids.cuda()
        encoder_input_attention_mask = encoder_input_attention_mask.cuda()

    num_beams = 8
    num_return_sequences = 8

    with torch.no_grad():
        model_outputs = model.generate(
            input_ids=encoder_input_ids,
            attention_mask=encoder_input_attention_mask,
            max_length=256,
            decoder_start_token_id=model.config.decoder_start_token_id,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences
        )

        model_outputs = model_outputs.view(1, num_return_sequences, model_outputs.shape[1])

    batch_inputs = [final_question]
    db_id = nat_table["db_id"]
    batch_db_ids = [db_id]
    table_dict = {db_id: nat_table}
    batch_tc_original = [nat_table['tc_fast']]

    predict_sqls = decode_natsqls(
        db_path,
        model_outputs,
        batch_db_ids,
        batch_inputs,
        tokenizer,
        batch_tc_original,
        table_dict
    )
    return predict_sqls


def transfer_spider_table_to_nat_table(tb_info):
    """
    tb_info是spider格式，
    只能是单个问题所涉及的表
    必须是同一个DB下的表
    """
    dbs = transfer_table_info([tb_info])
    if len(dbs) != 1:
        raise Exception(f"Transfer spider tables error: db count - {len(dbs)}")

    return dbs[0]


def transfer_table_to_spider_table():
    raise NotImplemented('Please implement')


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,  # 设置日志级别
        format='%(asctime)s [%(levelname)s] %(message)s',  # 设置日志消息格式
        datefmt='%Y-%m-%d %H:%M:%S'  # 设置日期时间格式
    )

    # os.environ['T2S_RESDSQL_DEBUG'] = '1'
    os.environ['T2S_RESDSQL_MODEL'] = './models/text2natsql-mt5-base-cspider/checkpoint-32448'
    os.environ['T2S_DB_PATH'] = './database'
    # 将问题涉及到的表均加入
    test_tb_infos = {
        "db_id": "my_kk",
        "table_names": [
            "交管案件数据_城区"
        ],
        "column_names": [
            [-1, "*"], [0, "申请时间"], [0, "当事人电话"], [0, "问题内容"], [0, "问题分类1"], [0, "问题分类"],
            [0, "来源分类"], [0, "问题分类2"], [0, "行政区划"], [0, "流程时间"], [0, "案件地点"], [0, "道路名称"],
            [0, "处理人单位"], [0, "路段代码"], [0, "主键"], [0, "周"], [0, "时"]
        ],
        "table_names_original": [
            "交管案件数据_城区"
        ],
        "column_names_original": [
            [-1, "*"], [0, "申请时间"], [0, "当事人电话"], [0, "问题内容"], [0, "问题分类1"], [0, "问题分类"],
            [0, "来源分类"], [0, "问题分类2"], [0, "行政区划"], [0, "流程时间"], [0, "案件地点"], [0, "道路名称"],
            [0, "处理人单位"], [0, "路段代码"], [0, "主键"], [0, "周"], [0, "时"]
        ],
        "column_types": ["text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text", "text",
                         "text", "text", "text", "text", "text"],
        "foreign_keys": [],
        "primary_keys": []
    }
    nat_table = transfer_spider_table_to_nat_table(test_tb_infos)
    print(nat_table)

    res = predict_with_natsql('请查询各道路的交通秩序的案件数量? （there are values equal to "交通秩序" in `问题分类` column）', nat_table)
    print(res)
