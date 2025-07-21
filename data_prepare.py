import subprocess
from config import MYSQL_CONFIG, BASE_DIR

tag = {
    "error": "\033[1;31m[ERROR]\033[0m",
    "success": "\033[1;32m[SUCCESS]\033[0m",
    "processing": "\033[1;34m[PROCESSING]\033[0m",
}


# 导入数据到 MySQL
def create_database(mysql_config, sql_file_path):
    """使用 sql 文件创建数据库并导入数据"""

    # MySQL 命令前缀，包括 host、user、password
    mysql_cmd_prefix = [
        "mysql",
        "-h",
        "localhost",
        "-u",
        mysql_config["user"],
        f"-p{mysql_config['password']}",
    ]

    # 创建数据库
    print(f"{tag['processing']} 创建 {mysql_config['database']}")
    cmd = mysql_cmd_prefix + [
        "-e",
        f"DROP DATABASE IF EXISTS {mysql_config['database']}; CREATE DATABASE {mysql_config['database']};",
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        error_msg = result.stderr.splitlines()[1:][0]
        print(f"{tag['error']} {error_msg}")
        return
    print(f"{tag['success']} {mysql_config['database']} 创建成功")

    # 导入数据
    print(f"{tag['processing']} 导入数据")
    with open(sql_file_path, "r") as sql_file:
        result = subprocess.run(
            mysql_cmd_prefix + [mysql_config["database"]],
            stdin=sql_file,
            stderr=subprocess.PIPE,
            text=True,
        )
    if result.returncode != 0:
        error_msg = result.stderr.splitlines()[1:][0]
        print(f"{tag['error']} {error_msg}")
        return
    print(f"{tag['success']} 数据导入成功")


create_database(MYSQL_CONFIG, BASE_DIR / "data" / "region.sql")
