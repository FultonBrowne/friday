import logging
import string
from numpy import append
import torch
from torchtext.data import get_tokenizer
from tokenizers import Tokenizer
from tokenizers.models import BPE
from transformers import AutoTokenizer

class TextTools():
    def __init__(self, tokenizer="basic_english") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


    def collate_tokenize(self, data):
        text_batch = [element["data"] for element in data]
        dtokenized = self.tokenizer(text_batch, padding='longest', truncation=True, return_tensors = 'pt')
        text_batch = [element["target"] for element in data]
        ttokenized = self.tokenizer(text_batch, padding='longest', truncation=True, return_tensors = 'pt')
        return {r'data': dtokenized.input_ids, r'target': ttokenized}


def text_dataset_bytetoken(input):
    t = TextTools()
    output = []
    for i in input:
        print(type(i))
        output.append(t.token_bytes(i))
    return output
def test_text_tokenizer():
    input_text = "Hello, this is your pal friday, a digital voice assistant that can proccess Text, Images, Videos, Audio, and Datasets. abcdefghijklmnopqrstuvwrxyz ABCDEFGHIJKLMNOPQRSTUVWRXYZ 1234567890 ~!@#$%^&*()_+-=`[]{}\\<>,.?/:;"
    output_tokens = ['hello', ',', 'this', 'is', 'your', 'pal', 'friday', ',', 'a', 'digital', 'voice', 'assistant', 'that', 'can', 'proccess', 'text', ',', 'images', ',', 'videos', ',', 'audio', ',', 'and', 'datasets', '.', 'abcdefghijklmnopqrstuvwrxyz', 'abcdefghijklmnopqrstuvwrxyz', '1234567890', '~', '!', '@#$%^&*', '(', ')', '_+-=`[]{}\\<>', ',', '.', '?', '/']
    t = TextTools()
    assert t.get_tokens(input_text) == output_tokens
