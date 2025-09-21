import config
import pymysql
from rapidfuzz import fuzz
from token_classification import predict
from transformers import BertForTokenClassification, BertTokenizerFast


def address_alignment(text: str, model, tokenizer, label_name, mysql_config) -> dict:
    """地址对齐"""
    # 序列标注
    tagging = predict(text, model, tokenizer, label_name)
    # 提取各级别地址
    address = address_extract(text, tagging)
    # 校验地址信息
    correct_address = address_check(text, address, mysql_config)
    return {
        "省份": correct_address[2],
        "城市": correct_address[3],
        "区县": correct_address[4],
        "街道": correct_address[5],
        "详细地址": address[6],
    }


def address_extract(text: str, tagging: list[str]) -> dict[int, str]:
    """地址提取"""
    tagging = [i[2:] for i in tagging]
    # 标签到地区类别 id 映射表
    label_map = {
        "": 0,
        "prov": 2,
        "city": 3,
        "district": 4,
        "road": 5,
        "detail": 6,
    }
    # 各级别对应地址信息
    address = {2: None, 3: None, 4: None, 5: None, 6: None}
    # 提取出各级别地址信息
    start_pos = 0
    tag_len = len(tagging)
    for end_pos in range(tag_len):
        # 如果到结尾、或 end_pos 的下一个位置不是同一类
        if (end_pos == tag_len - 1) or (tagging[end_pos + 1] != tagging[start_pos]):
            if tagging[start_pos] != "":
                address[label_map[tagging[start_pos]]] = text[start_pos : end_pos + 1]
            start_pos = end_pos + 1
    return address


def address_check(text: str, address: dict[int, str], mysql_config) -> dict[int, str]:
    """地址校验"""

    def flatten_address_tree(tree, chain=[]):
        """将树展平"""
        chains = []
        # 非叶节点
        if isinstance(tree, dict) and tree:
            for k, v in tree.items():
                chains.extend(flatten_address_tree(v, chain + [k]))
        # 叶节点
        elif isinstance(tree, list) and tree:
            for road in tree:
                chains.append(chain + [road])
        else:
            chains.append(chain)
        return chains

    region_types = [2, 3, 4, 5]
    with pymysql.connect(**mysql_config) as mysql_conn:
        with mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
            params_list = [(k, address[k]) for k in region_types]
            # 处理 省 市 区 标签错位
            params_list.extend([(2, address[3]), (3, address[2]), (3, address[4])])
            # 过滤空值
            params_list = [(i[0], f"%{i[1]}%") for i in params_list if i[1]]

            # 将多个(region_type=%s, name like %s)条件用or拼接,查询每个层级对应的full_name
            placeholders = "(region_type=%s and name like %s)"
            sql = (
                f"select full_name from region where {placeholders}"
                + f" or {placeholders}" * (len(params_list) - 1)
            )
            params = [i for pair in params_list for i in pair]
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            prefixes = [row["full_name"].split(" ") for row in rows]

    # 合并前缀
    # {
    #     "prov": {
    #         "city": {
    #             "dist": ["road"],
    #         },
    #     },
    # }
    address_tree = {}
    leaf_id = region_types[-1] - region_types[0]
    for prefix in prefixes:
        node = address_tree
        for i in range(len(prefix)):
            if i == leaf_id:
                node.append(prefix[i])
            elif i == leaf_id - 1:
                node = node.setdefault(prefix[i], [])
            else:
                node = node.setdefault(prefix[i], {})

    # 展平
    # [["prov", "city", "dist", "road"]]
    address_chains = flatten_address_tree(address_tree)

    # 转换为地址文本
    address_texts = ["".join(i) for i in address_chains]

    # 基于编辑距离计算两字符串的相似度
    # 编辑距离用来衡量两个字符串之间相似度
    # 表示从一个字符串A变成另一个字符串B，所需要的最少单字符编辑操作次数
    # 编辑操作包括插入、删除、替换
    scores = [fuzz.ratio(i, text) for i in address_texts]

    # 取分数最高的结果
    correct_address = {k: None for k in region_types}
    correct_address_chain = address_chains[scores.index(max(scores))]
    correct_address.update(zip(region_types, correct_address_chain))

    return correct_address


if __name__ == "__main__":
    model_path = config.FINETUNED_PATH / "best"
    model = BertForTokenClassification.from_pretrained(model_path).to(config.DEVICE)
    tokenizer = BertTokenizerFast.from_pretrained(model_path)
    text = [
        "中国浙江省杭州市余杭区葛墩路27号楼",
        "北京市市辖区通州区永乐店镇27号楼",
        "北京市市辖区东风街道27号楼",
        "新疆维吾尔自治区划阿拉尔市金杨镇27号楼",
        "甘肃省南市文县碧口镇27号楼",
        "陕西省渭南市华阴市罗镇27号楼",
        "西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        "广州市花都区花东镇27号楼",
    ]
    for i in text:
        address_dict = address_alignment(
            i, model, tokenizer, config.LABELS, config.MYSQL_CONFIG
        )
        print(address_dict)
