import torch
from pathlib import Path

MYSQL_CONFIG = {
    "host": "127.0.0.1",
    "port": 3306,
    "user": "root",
    "password": "123321",
    "database": "region",
    "charset": "utf8mb4",
}

# SQl文件路径
SQL_FILE_PATH = Path("data/region.sql")
# 原始数据路径
RAW_DATA_PATH = Path("data/raw")
# 已处理数据存放路径
PROCESSED_DATA_PATH = Path("data/processed")
# 模型参数保存路径
FINETUNED_PATH = Path("finetuned")
# TensorBoard 日志保存路径
LOGS_PATH = Path("logs")
# 本地预训练模型路径
PRETRAINED_PATH = Path("~/models").expanduser()
BERT_BASE = PRETRAINED_PATH / "bert-base-chinese"
ROBERTA_SMALL = PRETRAINED_PATH / "roberta-small-wwm-chinese-cluecorpussmall"

# 设备
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 标签
RAW_LABELS = [
    "O",
    "B-prov",
    "I-prov",
    "E-prov",
    "B-city",
    "I-city",
    "E-city",
    "B-district",
    "I-district",
    "S-district",
    "E-district",
    "B-road",
    "I-road",
    "E-road",
    "B-assist",
    "I-assist",
    "S-assist",
    "E-assist",
    "B-cellno",
    "I-cellno",
    "E-cellno",
    "B-community",
    "I-community",
    "S-community",
    "E-community",
    "B-devzone",
    "I-devzone",
    "E-devzone",
    "B-floorno",
    "I-floorno",
    "E-floorno",
    "B-houseno",
    "I-houseno",
    "E-houseno",
    "B-poi",
    "I-poi",
    "S-poi",
    "E-poi",
    "B-roadno",
    "I-roadno",
    "E-roadno",
    "B-subpoi",
    "I-subpoi",
    "E-subpoi",
    "B-town",
    "I-town",
    "E-town",
    "B-intersection",
    "I-intersection",
    "S-intersection",
    "E-intersection",
    "B-distance",
    "I-distance",
    "E-distance",
    "B-village_group",
    "I-village_group",
    "E-village_group",
]

LABELE_MAP = {
    "": "",
    "prov": "prov",
    "city": "city",
    "district": "district",
    "road": "road",
    "intersection": "road",
    "town": "road",
    "roadno": "detail",
    "cellno": "detail",
    "community": "detail",
    "houseno": "detail",
    "poi": "detail",
    "subpoi": "detail",
    "assist": "detail",
    "distance": "detail",
    "village_group": "detail",
    "floorno": "detail",
    "devzone": "detail",
}

LABELS = [
    "O",
    "B-prov",
    "I-prov",
    "E-prov",
    "B-city",
    "I-city",
    "E-city",
    "B-district",
    "I-district",
    "E-district",
    "S-district",
    "B-road",
    "I-road",
    "E-road",
    "S-road",
    "B-detail",
    "I-detail",
    "E-detail",
    "S-detail",
]
