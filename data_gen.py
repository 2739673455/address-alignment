import os
import json
import config
from pathlib import Path
from openai import OpenAI
from threading import Thread
from jinja2 import Environment, FileSystemLoader

client = OpenAI(
    api_key=os.getenv("KIMI_API_KEY"),
    base_url="https://api.moonshot.cn/v1",
)


template_dir = Path(__file__).parent
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("prompt.jinja2")
with (
    open(config.RAW_DATA_PATH / "data.jsonl", "r") as r_f,
    open(config.RAW_DATA_PATH / "correct_data.jsonl", "w") as w_f,
):
    all_data = [json.loads(line) for line in r_f]
    batch_size = 50
    for i in range(0, len(all_data), batch_size):
        datas = all_data[i : i + batch_size]
        prompt = template.render(datas=datas)
        completion = client.chat.completions.create(
            model="kimi-k2-turbo-preview",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=16384,
        )
        w_f.writelines(completion.choices[0].message.content + "\n")
        break
