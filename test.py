import os, sys

sys.path.append('../')
os.chdir('../')

import random
import numpy as np
import pandas as pd
import torch
from torch import optim
from tqdm import tqdm

from transformers import BertConfig, BertTokenizer
from nltk.tokenize import word_tokenize

from modules.word_classification import BertForWordClassification
from utils.forward_fn import forward_word_classification
from utils.metrics import ner_metrics_fn
from utils.data_utils import NerGritDataset, NerDataLoader


###
# common functions
###
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def count_param(module, trainable=False):
    if trainable:
        return sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in module.parameters())


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def metrics_to_string(metric_dict):
    string_list = []
    for key, value in metric_dict.items():
        string_list.append('{}:{:.2f}'.format(key, value))
    return ' '.join(string_list)


def word_subword_tokenize(sentence, tokenizer):
    # Add CLS token
    subwords = [tokenizer.cls_token_id]
    subword_to_word_indices = [-1]  # For CLS

    # Add subwords
    for word_idx, word in enumerate(sentence):
        subword_list = tokenizer.encode(word, add_special_tokens=False)
        subword_to_word_indices += [word_idx for i in range(len(subword_list))]
        subwords += subword_list

    # Add last SEP token
    subwords += [tokenizer.sep_token_id]
    subword_to_word_indices += [-1]

    return subwords, subword_to_word_indices


def predict(text):
    # Load Tokenizer and Config
    tokenizer = BertTokenizer.from_pretrained('indobenchmark/indobert-base-p1')
    config = BertConfig.from_pretrained('indobenchmark/indobert-base-p1')
    config.num_labels = NerGritDataset.NUM_LABELS

    # Instantiate model
    w2i, i2w = NerGritDataset.LABEL2INDEX, NerGritDataset.INDEX2LABEL
    model = BertForWordClassification.from_pretrained('indobenchmark/indobert-base-p1', config=config)

    text = word_tokenize(text)
    subwords, subword_to_word_indices = word_subword_tokenize(text, tokenizer)

    subwords = torch.LongTensor(subwords).view(1, -1).to(model.device)
    subword_to_word_indices = torch.LongTensor(subword_to_word_indices).view(1, -1).to(model.device)
    logits = model(subwords, subword_to_word_indices)[0]

    preds = torch.topk(logits, k=1, dim=-1)[1].squeeze().numpy()
    labels = [i2w[preds[i]] for i in range(len(preds))]

    return pd.DataFrame({'words': text, 'label': labels})


if __name__ == '__main__':
    # set text
    text = 'raya samb gede, 299 toko bb kids'

    print(predict(text))

    print(predict(text))

    print(predict(text))
