import logging
import string

from torchtext.data import get_tokenizer

class TextTools():
    def __init__(self, tokenizer="basic_english") -> None:
        logging.info("PRE: build tokenizer")
        self.tokenizer = get_tokenizer(tokenizer)
        logging.info("DONE: build tokenizer")
    def get_tokens(self, input:string):
        logging.info("PRE: get token")
        x = self.tokenizer(input)
        logging.info("DONE: get tokens")
        return x

def test_text_tokenizer():
    input_text = "Hello, this is your pal friday, a digital voice assistant that can proccess Text, Images, Videos, Audio, and Datasets. abcdefghijklmnopqrstuvwrxyz ABCDEFGHIJKLMNOPQRSTUVWRXYZ 1234567890 ~!@#$%^&*()_+-=`[]{}\\<>,.?/:;"
    output_tokens = ['hello', ',', 'this', 'is', 'your', 'pal', 'friday', ',', 'a', 'digital', 'voice', 'assistant', 'that', 'can', 'proccess', 'text', ',', 'images', ',', 'videos', ',', 'audio', ',', 'and', 'datasets', '.', 'abcdefghijklmnopqrstuvwrxyz', 'abcdefghijklmnopqrstuvwrxyz', '1234567890', '~', '!', '@#$%^&*', '(', ')', '_+-=`[]{}\\<>', ',', '.', '?', '/']
    t = TextTools()
    assert t.get_tokens(input_text) == output_tokens
