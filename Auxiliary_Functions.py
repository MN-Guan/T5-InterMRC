import time
import torch
import numpy as np
import random
import re
import collections
import string
import math

torch.set_printoptions(precision=2, sci_mode=False)

def time_since(start_time):
    duration = time.time() - start_time
    minute = duration // 60
    second = duration % 60
    return  '%2dm%2ds' % (minute, second)

def get_time_str():
    time_struct = time.localtime()
    time_str = '_' + str(time_struct.tm_mon) + '-' + str(time_struct.tm_mday) + '-' + str(time_struct.tm_hour)
    return time_str
# In[3]:


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True

def convert_to_single_gpu(state_dict):
    def _convert(key):
        if key.startswith('module.'):
            return key[7:]
        return key
    return {_convert(key): value for key, value in state_dict.items()}

def convert_optionstxt_to_list(optionstxt):
    """(A) one (B) two (C) three -> ['one', 'two', 'three']"""
    regex = re.compile("\([A-Z]\)")
    optionstxt_split = regex.split(optionstxt)
    options_list = [option.strip() for option in optionstxt_split if len(option.strip()) > 0]
    return options_list

def find_answer_in_options(answer, options):
    """
    input:
        answer: 'one'
        options:['one', 'two', 'three']
    return: 'A'
    """
    answer_index = options.index(answer.strip())
    return chr(ord('A') + answer_index)

def find_example_by_id(examples, qas_id):
    for example in examples:
        if example.qas_id == qas_id:
            return example

def make_acc_eval_dict(accuracy_list):
    total = len(accuracy_list)
    result = collections.OrderedDict(
        [
            ('accuracy', 100.0 * sum(accuracy_list) / total),
            ('total', total)
        ]
    )
    return result

def make_em_f1_eval_dict(exact_scores, f1_scores, qid_list=None):
    if not qid_list:
        total = len(exact_scores)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores.values()) / total),
                ("f1", 100.0 * sum(f1_scores.values()) / total),
                ("total", total),
            ]
        )
    else:
        total = len(qid_list)
        return collections.OrderedDict(
            [
                ("exact", 100.0 * sum(exact_scores[k] for k in qid_list) / total),
                ("f1", 100.0 * sum(f1_scores[k] for k in qid_list) / total),
                ("total", total),
            ]
        )


def string_to_float(string, default=-1.):
    try:
        return float(string)
    except ValueError:
        return default
   
def clean_label(label):
    num = label.count(3)
    for i in range(num):
        index = label.index(3)
        label.pop(index)
    return label
  
def compute_f1(a_gold, a_pred):
    gold_toks = a_gold.split()
    pred_toks = a_pred.split()
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def punish(differ, threshold):
    if differ <= threshold:
        return 0
    else:
        return 1

def loss_fct(pred_logits, labels, criterion):
    batch_len = pred_logits.shape[0]
    loss = torch.tensor(0., device=labels.device)
    total_len = 0
    if labels[0][0] == 0:
        # bart_label: [0, xxx, xxx]
        start_index = 1
    else:
        start_index = 0
    for i in range(batch_len):
        start_loss = criterion(pred_logits[i][start_index].view(-1, pred_logits.size(-1)), labels[i][start_index].view(-1))
        end_index = labels[i].tolist().index(1)
        end_loss = criterion(pred_logits[i][end_index].view(-1, pred_logits.size(-1)), labels[i][end_index].view(-1))
        sub_loss = criterion(pred_logits[i].view(-1, pred_logits.size(-1)), labels[i].view(-1)) + start_loss + end_loss
        # print(f'middle: {sub_loss.data}', end=' ---- ')
        label_len = labels[i].tolist().index(1) + 1
        total_len += label_len
        loss += sub_loss
    loss = loss / total_len   
    return loss


def loss_fct_old(pred_logits, labels, tokenizer, threshold, criterion):
    batch_len = pred_logits.shape[0]
    loss = torch.tensor(0.).to(labels.device)
    total_len = 0
    for i in range(batch_len):
        sub_loss = criterion(pred_logits[i].view(-1, pred_logits.size(-1)), labels[i].view(-1))
        pred = pred_logits[i].argmax(dim=-1)
        label_len = torch.count_nonzero(torch.where(labels[i]==-100, 0, 1))
        pred_tokens = tokenizer.decode(pred[:label_len-1], skip_special_tokens=True)
        pred_tokens = pred_tokens.replace(' ', '')
        label_tokens = tokenizer.decode(labels[i][:label_len-1], skip_special_tokens=True)
        label_tokens = label_tokens.replace(' ', '')
        total_len += label_len
        tokens_prob = torch.max(torch.nn.Softmax(dim=-1)(pred_logits[i][:label_len-1]), dim=-1).values
        probability = (tokens_prob[0] + tokens_prob[2]) / 2
        # print(torch.max(torch.nn.Softmax(dim=-1)(pred_logits[i][:label_len-1]), dim=-1).values)
        # print(probability.data, tokens_prob[0].data, tokens_prob[2].data)
        if string_to_float(pred_tokens) != -1:
            pred_tokens = float(pred_tokens)
            label_tokens = float(label_tokens)
            differ = abs(pred_tokens-label_tokens)
            if probability >= 0:
                sub_loss = punish(differ, threshold) * sub_loss
        loss += sub_loss
    loss = loss / total_len           
    # print('--------------')
    # loss2 = criterion(pred_logits.view(-1, pred_logits.size(-1)), labels.view(-1))
    return loss

