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
        completion = client.chat.completions.create(
            model="x-ai/grok-4-fast:free",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8192,
        )
        print(completion)
        w_f.write(completion.choices[0].message.content + "\n")
        break
