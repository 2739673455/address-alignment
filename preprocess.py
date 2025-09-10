import os
import random
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from datasets import Dataset, load_from_disk


class Processor:
    def __init__(
        self,
        data_path,
        save_dir,
        tokenizer,
        max_seq_len,
        train_ratio=0.8,
        test_ratio=0.1,
    ):
        self.data_path = data_path
        self.save_dir = save_dir
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio

    def process(self):
        """处理数据集并保存"""
        dataset = self._make_dataset()
        dataset = self._split_dataset(dataset)
        # 保存数据集
        for type in ["train", "valid", "test"]:
            dataset[type].save_to_disk(self.save_dir / type)

    def get_dataloader(self, type, batch_size, max_examples=None):
        """获取保存的数据集，并加载 Dataloader"""
        if not os.path.exists(self.save_dir / type):
            self.process()
        # 加载数据集
        dataset = load_from_disk(self.save_dir / type)
        # 如果限制了最大样本数，获取不多于 max_examples 的样本
        if max_examples:
            indices = list(range(len(dataset)))
            random.shuffle(indices)
            max_examples = min(max_examples, len(dataset))
            indices = indices[:max_examples]
            dataset = Subset(dataset, indices)
        return DataLoader(dataset, batch_size, shuffle=(type == "train"))

    def _make_dataset(self):
        """处理数据集"""
        raise NotImplementedError

    def _split_dataset(self, dataset):
        """划分数据集"""
        train_size = int(dataset.num_rows * self.train_ratio)
        dataset = dataset.train_test_split(test_size=self.test_ratio)
        train_valid_split = dataset["train"].train_test_split(train_size=train_size)
        dataset["train"] = train_valid_split["train"]
        dataset["valid"] = train_valid_split["test"]
        return dataset


class AddressTaggingProcessor(Processor):
    def __init__(
        self,
        data_path,
        save_dir,
        tokenizer,
        label_list,
        max_seq_len=64,
        train_ratio=0.8,
        test_ratio=0.1,
    ):
        super().__init__(
            data_path,
            save_dir,
            tokenizer,
            max_seq_len,
            train_ratio,
            test_ratio,
        )
        self.label_list = label_list
        self.label2id = {label: i for i, label in enumerate(self.label_list)}

    def _make_dataset(self):
        """处理数据集"""
        dataset = Dataset.from_generator(self._generate_examples)
        dataset = dataset.map(
            self._map_fn, batched=True, remove_columns=["text", "labels"]
        )
        dataset.set_format(
            type="torch", columns=["input_ids", "attention_mask", "labels"]
        )
        return dataset

    def _generate_examples(self):
        """
        将文本分块，每块是一个样本
        每块中每行是一个词和标签
        """
        with open(self.data_path, "r", encoding="utf-8") as f:
            blocks = f.read().split("\n\n")
            for block in blocks:
                text, labels = [], []
                lines = block.split("\n")
                for line in lines:
                    if not line.strip():
                        continue
                    word, label = line.strip().split()
                    text.append(word)
                    labels.append(self.label2id[label])
                yield {"text": text, "labels": labels}

    def _map_fn(self, examples):
        inputs = self.tokenizer(
            examples["text"],
            is_split_into_words=True,
            max_length=self.max_seq_len,
            truncation=True,
            padding="max_length",
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        labels = examples["labels"]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
