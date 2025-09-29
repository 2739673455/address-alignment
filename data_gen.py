import os
import json
import config
from pathlib import Path
from openai import OpenAI
from jinja2 import Environment, FileSystemLoader

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# 加载 prompt 模板
template_dir = Path(__file__).parent
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("prompt.jinja2")

with (
    open(config.RAW_DATA_PATH / "data.jsonl", "r") as r_f,
    open(config.RAW_DATA_PATH / "correct_data.jsonl", "a+") as w_f,
):
    # 获取已处理的行数
    w_f.seek(0)
    processed_line_num = len(w_f.readlines())

    batch_size = 20
    all_data = [json.loads(line) for line in r_f]
    for i in range(processed_line_num, len(all_data), batch_size):
        lines = "\n".join([str(d) for d in all_data[i : i + batch_size]])
        prompt = template.render(datas=lines)
        stream = client.chat.completions.create(
            model="deepseek/deepseek-chat-v3.1:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16384,
            stream=True,
        )
        completion = []
        for chunk in stream:
            delta = chunk.choices[0].delta
            if delta.content:
                completion.append(delta.content)
            print(delta.content, end="", flush=True)
        # 强制在最后添加换行符
        if not completion[-1].endswith("\n"):
            completion[-1] += "\n"
        # 将单引号替换为双引号
        w_f.write("".join(completion).replace("'", '"'))
        w_f.flush()
        print(f"\033[1;32m{i}:{i + batch_size} 已写入\033[0m")
