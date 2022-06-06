import torch
from transformers import BertTokenizer
english_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
chinese_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
if __name__ == '__main__':
    sentence = 'hello my name is zhang '
    c = ['你好呀','我好呀']
    print(chinese_tokenizer(c,padding='max_length'))
