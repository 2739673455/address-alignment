import config
from datetime import datetime
from train import AddressTaggingTrainer
from preprocess import AddressTaggingProcessor
from models_def import AddressTagging, load_params
from transformers import BertForTokenClassification
from torch.utils.tensorboard.writer import SummaryWriter

batch_size = 16
learning_rate = 1e-5
device = config.DEVICE


def model_go(train=0, test=0, inference=0, model_params_path=None):
    model = AddressTagging(config.BERT_MODEL, config.LABELS)
    processor = AddressTaggingProcessor(
        data_path=config.RAW_DATA_DIR / "data.txt",
        save_dir=config.PROCESSED_DATA_DIR,
        tokenizer=model.tokenizer,
        label_list=config.LABELS,
    )
    trainer = AddressTaggingTrainer(model, device, 10, learning_rate)

    save_name = f"address_tagging-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    load_params(model, model_params_path)
    writer = None

    if train:
        writer = SummaryWriter(config.LOGS_DIR / save_name)
        dataloader = {
            "train": processor.get_dataloader("train", batch_size),
            "valid": processor.get_dataloader("valid", batch_size),
        }
        model_params_path = config.FINETUNED_DIR / f"{save_name}.pt"
        trainer(dataloader, model_params_path, writer)

    if test:
        test_dataloader = processor.get_dataloader("test", batch_size)
        trainer({"test": test_dataloader}, writer=writer, is_test=True)

    if writer:
        writer.close()

    if inference:
        res = model.predict(text, device)
        if isinstance(res, str):
            print(res)
        elif isinstance(res, list):
            for a_text, a_res in zip(text, res):
                for t, r in zip(a_text, a_res):
                    print(f"{t}\t{r}", end="\n")


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

model_go(0, 0, 1, config.FINETUNED_DIR / "address_tagging.pt")
