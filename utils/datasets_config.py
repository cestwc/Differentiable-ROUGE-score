from datasets import load_dataset
from utils import tokenizer_config

from datasets import load_dataset


def get_dataset(name, tokenize_algorithm, set_format = True):
    sample_tokenizer = tokenizer_config.SampleTokenizer(tokenize_algorithm)
    if name == 'gigaword':
        dset = load_dataset('gigaword').map(sample_tokenizer.sample_tokenize, batched=True)
    elif name == 'cnn_dailymail':
        dset = load_dataset('ccdv/cnn_dailymail', '3.0.0').map(sample_tokenizer.sample_tokenize, batched=True)
        dset['train'] = dset['train'].filter(lambda x: (len(x['labels']) > 10))

    print(f"Number of {name} training examples: {len(dset['train'])}")
    print(f"Number of {name} validation examples: {len(dset['validation'])}")
    print(f"Number of {name} testing examples: {len(dset['test'])}")

    if set_format:
        dset.set_format(type='torch', columns=['input_ids','labels', 'attention_mask'])
    return dset