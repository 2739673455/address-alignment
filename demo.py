import config
from transformers import BertForTokenClassification, BertModel

# model = BertForTokenClassification.from_pretrained(config.BERT_MODEL)
model = BertModel.from_pretrained(config.BERT_MODEL)
print(model)
