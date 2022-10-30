from transformers import AutoTokenizer
import torch
import re

def remove_punct(text):
    w = re.sub(r'[^\w\s#@\_]',' ',text) 
    return w
    
class SampleTokenizer:
    def __init__(self, pretrained_model_name_or_path):
        self.sample_tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast = True)

    def sample_tokenize(self, e):
        if 'document' in e:
            sample = self.sample_tokenizer([remove_punct(x) for x in e['document']], max_length=512, truncation=True)
            sample['labels'] = self.sample_tokenizer([remove_punct(x) for x in e['summary']], max_length=512, truncation=True)['input_ids']
            sample['input_texts'] = e['document']
            del e['document']
            sample['label_texts'] = e['summary']
            del e['summary']
        else:
            sample = self.sample_tokenizer([remove_punct(x) for x in e['article']], max_length=512, truncation=True)
            sample['labels'] = self.sample_tokenizer([remove_punct(x) for x in e['highlights']], max_length=512, truncation=True)['input_ids']
            sample['input_texts'] = e['article']
            del e['article']
            sample['label_texts'] = e['highlights']
            del e['highlights']
        return sample
