import os
import json
import config
import asyncio
import functools
from pathlib import Path
from typing import Callable
from openai import AsyncOpenAI
from rich.console import Console
from jinja2 import Environment, FileSystemLoader

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

# 加载 prompt 模板
template_dir = Path(__file__).parent
env = Environment(loader=FileSystemLoader(template_dir))
template = env.get_template("prompt.jinja2")

console = Console()


class MaxRetriesError(Exception):
    pass


def with_retry(max_retries: int, delay: float):
    """重试装饰器"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            for retry in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    console.print(f"第 {retry + 1} 次尝试失败: {e}", style="red")
                    if retry == max_retries - 1:
                        raise MaxRetriesError(f"达到最大重试次数: {max_retries}")
                    await asyncio.sleep(delay)
            return None

        return async_wrapper

    return decorator


def with_valid():
    """验证装饰器"""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            batch_data = args[0] if args else kwargs.get("batch_data", [])
            result = await func(*args, **kwargs)
            # 数量验证
            if len(result) != len(batch_data):
                raise Exception("数据数量不匹配")
            # 内容验证
            for original, processed in zip(batch_data, result):
                if original["text"] != processed.get("text"):
                    raise Exception("数据内容不匹配")
            return result

        return async_wrapper

    return decorator


@with_retry(max_retries=5, delay=2.0)
@with_valid()
async def process_batch_data(batch_data: list[dict]) -> list[dict]:
    """处理一批数据"""
    text = "\n".join([str(d) for d in batch_data])
    prompt = template.render(datas=text)
    stream = await client.chat.completions.create(
        model="deepseek/deepseek-chat-v3.1:free",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=16384,
        stream=True,
    )
    # 流式输出
    completion = []
    async for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            completion.append(delta.content)
    completion = "".join(completion).replace("'", '"')  # 将单引号替换为双引号
    completion_datas = [json.loads(line) for line in completion.split("\n")]
    return completion_datas


class OrderedWriter:
    def __init__(self, output_file, cur_line_num):
        self.output_file = output_file
        self.completed_batches = {}
        self.cur_line_num = cur_line_num

    async def add_completed_batch(self, i):
        """写入文件"""
        while True:
            # 获取当前待写入的批次数据
            cur_batches = self.completed_batches.get(self.cur_line_num)
            print(
                f"待写入 {self.cur_line_num}，当前序号 {list(self.completed_batches.keys())}"
            )
            # 如果不存在则返回
            if not cur_batches:
                break
            # 写入这一批次的数据
            for i in cur_batches:
                self.output_file.write(json.dumps(i, ensure_ascii=False) + "\n")
            self.output_file.flush()
            del self.completed_batches[self.cur_line_num]
            # 更新待写入数据的行号
            self.cur_line_num += len(cur_batches)
            console.print(
                f"{self.cur_line_num}:{self.cur_line_num + len(cur_batches)} 已写入",
                style="green",
            )


async def process_single_batch(i, batch_datas, writer: OrderedWriter):
    console.print(f"{i}:{i + len(batch_datas)} 处理中", style="blue")
    completion_datas = await process_batch_data(batch_datas)
    writer.completed_batches[i] = completion_datas
    await writer.add_completed_batch(i)


async def main():
    with open(config.RAW_DATA_PATH / "data.jsonl", "r") as r_f:
        all_data = [json.loads(line) for line in r_f]  # 读取所有数据
    with open(config.RAW_DATA_PATH / "correct_data.jsonl", "a+") as w_f:
        w_f.seek(0)  # 移动到文件开头
        processed_line_num = len(w_f.readlines())  # 获取已处理的行数
        writer = OrderedWriter(w_f, processed_line_num)
        batch_size = 20  # 每批次处理的数量
        tasks = []  # 任务列表
        for i in range(processed_line_num, len(all_data), batch_size):
            batch_datas = all_data[i : i + batch_size]
            tasks.append(
                asyncio.create_task(process_single_batch(i, batch_datas, writer))
            )
            if len(tasks) >= 5 or i + batch_size >= len(all_data):
                await asyncio.gather(*tasks, return_exceptions=True)
                tasks.clear()


asyncio.run(main())
