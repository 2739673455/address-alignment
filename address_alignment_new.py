import config
import pymysql
from difflib import SequenceMatcher
from models_def import AddressTagging, load_params


def address_alignment(text: str, model: AddressTagging) -> dict:
    # 获取标注结果
    tagging = model.predict(text)
    # 填充各级别地址
    address = address_extract(text, tagging)
    # 校验地址信息
    correct_address = check_address(address)

    print("原始结果:", address)
    print("校验结果:", correct_address)

    return {
        "省份": correct_address[2],
        "城市": correct_address[3],
        "区县": correct_address[4],
        "街道": correct_address[5],
        "详细地址": address[6],
    }


def address_extract(text: str, tagging: list[str]) -> dict[int, str]:
    # 标签到地区类别 id 映射表
    label_map = {
        "": 0,
        "prov": 2,
        "city": 3,
        "district": 4,
        "road": 5,
        "intersection": 5,
        "town": 5,
        "roadno": 6,
        "cellno": 6,
        "community": 6,
        "houseno": 6,
        "poi": 6,
        "subpoi": 6,
        "assist": 6,
        "distance": 6,
        "village_group": 6,
        "floorno": 6,
        "devzone": 6,
    }
    # 各级别对应地址信息
    address = {2: None, 3: None, 4: None, 5: None, 6: None}
    # 提取出各级别地址信息
    start_pos = 0
    tag_len = len(tagging)
    for end_pos in range(tag_len):
        # 如果到结尾、或 end_pos 的下一个位置不是同一类
        if (end_pos == tag_len - 1) or (
            label_map[tagging[end_pos + 1]] != label_map[tagging[start_pos]]
        ):
            if label_map[tagging[start_pos]] != 0:
                # 添加片段
                address[label_map[tagging[start_pos]]] = text[start_pos : end_pos + 1]
            start_pos = end_pos + 1
    return address


def check_address(address: dict[int, str]) -> dict[int, str]:
    """校验地址"""

    def flatten_address_tree(tree, chain=[]):
        """将树展平成地址链"""
        chains = []
        # 非叶节点
        if isinstance(tree, dict):
            for k, v in tree.items():
                chains.extend(flatten_address_tree(v, chain + [k]))
        # 叶节点
        elif isinstance(tree, list):
            if tree:
                for road in tree:
                    chains.append(chain + [road])
            else:
                chains.append(chain)
        return chains

    address_tree = {}
    with pymysql.connect(**config.MYSQL_CONFIG) as mysql_conn:
        with mysql_conn.cursor(pymysql.cursors.DictCursor) as cursor:
            region_type_ids = [2, 3, 4, 5]
            for region_type_id in region_type_ids:
                # 查出每个级别符合条件的前缀
                sql = (
                    "select full_name from region where region_type=%s and name like %s"
                )
                params = (region_type_id, f"%{address[region_type_id]}%")
                cursor.execute(sql, params)
                prefixes = [i["full_name"].split(" ") for i in cursor.fetchall()]

                # 合并前缀，构造成地址树
                # {
                #     "prov": {
                #         "city": {
                #             "dist": ["road"],
                #         },
                #     },
                # }
                leaf_id = region_type_ids[-1] - region_type_ids[0]
                for prefix in prefixes:
                    node = address_tree
                    for i in range(len(prefix)):
                        if i == leaf_id:
                            node.append(prefix[i])
                        elif i == leaf_id - 1:
                            node = node.setdefault(prefix[i], [])
                        else:
                            node = node.setdefault(prefix[i], {})

    # 转换为地址链
    # [["prov", "city", "dist", "road"]]
    address_chains = flatten_address_tree(address_tree)

    # 转换为地址文本
    address_texts = ["".join(i) for i in address_chains]
    raw_address_text = "".join([address[i] for i in region_type_ids if address[i]])

    # 计算编辑距离
    scores = []
    for i in address_texts:
        scores.append(SequenceMatcher(None, i, raw_address_text).ratio())

    # 取分数最高的结果
    correct_address = {}
    for i in region_type_ids:
        correct_address_chain = address_chains[scores.index(max(scores))]
        correct_address[i] = correct_address_chain[i - region_type_ids[0]]
    return correct_address


if __name__ == "__main__":
    model = AddressTagging(config.BERT_MODEL, config.LABELS)
    load_params(model, config.FINETUNED_DIR / "address_tagging.pt")
    text = [
        # "中国浙江省杭州市余杭区葛墩路27号楼",
        # "北京市通州区永乐店镇27号楼",
        "北京市市辖区高地街道27号楼",
        # "新疆维吾尔自治区划阿拉尔市金杨镇27号楼",
        # "甘肃省南市文县碧口镇27号楼",
        # "陕西省渭南市华阴市罗镇27号楼",
        # "西藏自治区拉萨市墨竹工卡县工卡镇27号楼",
        # "广州市花都区花东镇27号楼",
    ]
    for i in text:
        print(address_alignment(i, model))
