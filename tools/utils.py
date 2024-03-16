import argparse
from collections import Counter
import re
import numpy as np
from torch import Tensor
import torch
import torch.nn as nn
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk import word_tokenize
from typing import Dict

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def ks_to_cuda(batch, device):
    input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask, gt, word_node_gt = batch
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    token_type_ids = token_type_ids.to(device)
    know_ids = know_ids.to(device)
    graph = graph.to(device)
    node_mask = node_mask.to(device)
    word_node_gt = word_node_gt.to(device)
    gt = gt.to(device)
    bs, _, _ = input_ids.size()
    for j in range(bs):
        for h in range(len(ent_node_embed[j])):
            ent_node_embed[j][h][0] = ent_node_embed[j][h][0].to(device)
            ent_node_embed[j][h][1] = ent_node_embed[j][h][1].to(device)
    return input_ids, attention_mask, token_type_ids, graph, ent_node_embed, know_ids, node_mask, gt, word_node_gt


def gpt_to_cuda(batch, device):
    # input_ids, attention_mask, token_type_ids, response_ids, response = batch
    #
    # input_ids = input_ids.to(device)
    # attention_mask = attention_mask.to(device)
    # token_type_ids = token_type_ids.to(device)
    # response_ids = response_ids.to(device)
    batch = list(batch)
    for i in range(len(batch)):
        if isinstance(batch[i], Tensor):
            batch[i] = batch[i].to(device)
    return batch


def charge_multi_label(target, preds):
    res = []
    for i in range(len(preds)):
        # res.append(sum(target[i]) // len(target[i]))
        if preds[i] in target[i]:
            res.append(preds[i])
        else:
            res.append(preds[i] + 1)

    return np.array(res)


def mark_only_lora_as_trainable(model: nn.Module, bias: str = 'none') -> None:
    
    for n, p in model.named_parameters():
        if 'lora_' not in n:
            p.requires_grad = False
        else:
            p.requires_grad = True
    if bias == 'none':
        return
    elif bias == 'all':
        for n, p in model.named_parameters():
            if 'bias' in n:
                p.requires_grad = True
    elif bias == 'lora_only':
        pass
    else:
        raise NotImplementedError


def lora_state_dict(model: nn.Module, bias: str = 'none') -> Dict[str, torch.Tensor]:
    my_state_dict = model.state_dict()
    if bias == 'none':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k}
    elif bias == 'all':
        return {k: my_state_dict[k] for k in my_state_dict if 'lora_' in k or 'bias' in k}
    elif bias == 'lora_only':
        to_return = {}
        for k in my_state_dict:
            if 'lora_' in k:
                to_return[k] = my_state_dict[k]
                bias_name = k.split('lora_')[0]+'bias'
                if bias_name in my_state_dict:
                    to_return[bias_name] = my_state_dict[bias_name]
        return to_return
    else:
        raise NotImplementedError


def remove_punctuation_and_words(text):
    text_no_punct = re.sub(r'[^\w\s]', '', text)
    
    words_to_remove = ['an', 'a', 'the']
    words_pattern = '\\b(?:' + '|'.join(words_to_remove) + ')\\b'
    text_no_words = re.sub(words_pattern, '', text_no_punct, flags=re.IGNORECASE)
    
    return text_no_words

def compute_bleu(pred, gold):
    def _preproc_preds_golds(pred, gold=None):
            cands = []
            golds = []
            help_tokenize = lambda x: word_tokenize(x.lower())
            for idx, p in enumerate(pred):
                cands.append(help_tokenize(p.lower()))
                if gold is not None:
                    golds.append(help_tokenize(gold[idx].lower()))
            return cands, golds

    hypothesis, references = _preproc_preds_golds(pred, gold)
    references = [[ref] for ref in references]
    sf = SmoothingFunction(epsilon=1e-12).method1
    b1 = corpus_bleu(references, hypothesis, weights=(1.0/1.0,), smoothing_function=sf)
    b2 = corpus_bleu(references, hypothesis, weights=(1.0/2.0, 1.0/2.0), smoothing_function=sf)
    b3 = corpus_bleu(references, hypothesis, weights=(1.0/3.0, 1.0/3.0, 1.0/3.0), smoothing_function=sf)
    b4 = corpus_bleu(references, hypothesis, weights=(1.0/4.0, 1.0/4.0, 1.0/4.0, 1.0/4.0), smoothing_function=sf)
    return {'BLEU-1': b1, 'BLEU-2': b2, 'BLEU-3': b3, 'BLEU-4': b4}


re_art = re.compile(r'\b(a|an|the)\b')	
re_punc = re.compile(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']')	

def normalize_answer(s):	
    """	
    Lower text and remove punctuation, articles and extra whitespace.	
    """	
    s = s.lower()	
    s = re_punc.sub(' ', s)	
    s = re_art.sub(' ', s)	
    # TODO: this could almost certainly be faster with a regex \s+ -> ' '	
    s = ' '.join(s.split())	
    return s.split(' ')

def _prec_recall_f1_score(pred_items, gold_items):
        """
        PARLAI
        Computes precision, recall and f1 given a set of gold and prediction items.
        :param pred_items: iterable of predicted values
        :param gold_items: iterable of gold values
        :return: tuple (p, r, f1) for precision, recall, f1
        """
        common = Counter(gold_items) & Counter(pred_items)
        num_same = sum(common.values())
        if num_same == 0:
            return 0, 0, 0
        precision = 1.0 * num_same / len(pred_items)
        recall = 1.0 * num_same / len(gold_items)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1, precision, recall

def get_unigram_F1(pred, gold):
    f1, precision, recall = [], [], []
    for p,g in zip(pred,gold):
        p = normalize_answer(p)
        g = normalize_answer(g)
        f1_i, precision_i, recall_i = _prec_recall_f1_score(p, g)

        f1.append(f1_i)
        precision.append(precision_i)
        recall.append(recall_i)
    return np.mean(f1), np.mean(precision), np.mean(recall)