import os
import json
import re
import numpy as np
import pandas as pd
import tqdm
import jsonlines
import sys

from transformers.tokenization_utils_base import PreTrainedTokenizerBase, TruncationStrategy
from transformers import T5Tokenizer, BertTokenizer, RobertaTokenizer
from Auxiliary_Functions import clean_label

def _string_join(lst):
    inputs = ' '.join(lst)
    regex_contiguous_space = re.compile(r'\s+')
    inputs = re.sub(regex_contiguous_space, ' ', inputs)
    return inputs

class STS_example():
    def __init__(self, sentence1, sentence2, gold_probability, sts_id):
        self.sentence1 = sentence1
        self.sentence2 = sentence2
        self.gold_probability = gold_probability
        self.sts_id = sts_id
        
    def print_content(self):
        print(f'sts_id: {self.sts_id}')
        print(f'sentence1: {self.sentence1}')
        print(f'sentence2: {self.sentence2}')
        print(f'gold_probability: {self.gold_probability}', end='\n\n')
        
def get_STS_examples(data_name, file_path, file_type, interval, do_augment):
    data_file = os.path.join(file_path, data_name)
    if data_name == 'SICK':
        SICK_data = pd.read_table(os.path.join(data_file, 'SICK.txt'), encoding='utf-8')
        type_map = {'train': 'TRAIN', 'dev': 'TRIAL', 'test': 'TEST'}
        data = SICK_data.loc[SICK_data['SemEval_set']==type_map[file_type], :]
        left_margin = 1
    elif data_name == 'STSb':
        file_name = os.path.join(data_file, f'sts-{file_type}.csv')
        data = pd.read_table(file_name, sep=r'\t', on_bad_lines='warn', engine='python', header=None)
        left_margin = 0
    elif data_name == 'STS':
        file_name = os.path.join(data_file, f'{file_type}.csv')
        data = pd.read_table(file_name, sep=r'\t', on_bad_lines='warn', engine='python', header=0)
        left_margin = 0
    else:
        raise ValueError("data_name must in ['SICK', 'STSb', 'STS']")

    tqdm_lines = tqdm.tqdm(range(data.shape[0]), desc=f'Obtain {data_name} {file_type} examples')
    examples = []
    sts_index = 0
    for line_index in tqdm_lines:
        if data_name == 'SICK':
            sentence1 = data.iloc[line_index, 1]   # sentence_A
            sentence2 = data.iloc[line_index, 2]   # sentence_B
            gold_probability = data.iloc[line_index, 4] # relatedness_score
        elif data_name == 'STSb':
            sentence1 = data.iloc[line_index, 5]   # sentence_A
            sentence2 = data.iloc[line_index, 6]   # sentence_B
            gold_probability = data.iloc[line_index, 4]  # relatedness_score
        else:
            sentence1 = data.loc[line_index, 'sentence1']   
            sentence2 = data.loc[line_index, 'sentence2']   
            gold_probability = data.loc[line_index, 'score'] 

        if file_type == 'train':
            if interval == 0.2:
                gold_probability = round(gold_probability * 5) / 5
                if gold_probability == left_margin or gold_probability == 5:
                    left_probability = None
                    right_probability = None
                else:
                    left_probability = round(gold_probability - 0.2, 1) if gold_probability - 0.2 >= left_margin else None
                    right_probability = round(gold_probability + 0.2, 1) if gold_probability + 0.2 <= 5 else None
            elif interval == 0.1:
                gold_probability = round(gold_probability, 1)
                if gold_probability == left_margin or gold_probability == 5:
                    left_probability = None
                    right_probability = None
                else:
                    left_probability = round(gold_probability - 0.1, 1) if gold_probability - 0.1 >= left_margin else None
                    right_probability = round(gold_probability + 0.1, 1) if gold_probability + 0.1 <= 5 else None
            else:
                gold_probability = gold_probability
                left_probability = None
                right_probability = None

                # raise ValueError('interval must be 0.1 or 0.2!')
        sts_id = f'{data_name}_' + file_type + str(sts_index)
        sts_index += 1
        examples.append(STS_example(sentence1, sentence2, gold_probability, sts_id))
        if file_type == 'train' and do_augment == True:
            # sts_id = f'{data_name}_' + file_type + str(sts_index)
            # sts_index += 1
            # examples.append(STS_example(sentence2, sentence1, gold_probability, sts_id))

            if left_probability != None:
                sts_id = f'{data_name}_' + file_type + str(sts_index)
                sts_index += 1
                examples.append(STS_example(sentence1, sentence2, left_probability, sts_id))
            if right_probability != None:
                sts_id = f'{data_name}_' + file_type + str(sts_index)
                sts_index += 1
                examples.append(STS_example(sentence1, sentence2, right_probability, sts_id))
    return examples


def STSFeatures(input_ids = None,
        attention_mask = None,
        token_type_ids = None,
        labels = None, 
        gold_probability = None,
        sts_id = None
    ):
        feature = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'token_type_ids': token_type_ids,
        'labels': labels, 
        'gold_probability': gold_probability,
        'sts_id': sts_id
        }
        return feature

def STS_convert_example_to_features(example, tokenizer, max_seq_length, is_training, do_divide):
    sentence1 = example.sentence1
    sentence2 = example.sentence2
    gold_probability  = example.gold_probability
    sts_id = example.sts_id
    assert isinstance(tokenizer, T5Tokenizer), 'The tokenizer used for generation preprocessing must be member of the T5Tokenizer'
    inputs = _string_join(['sts sentence1:', sentence1, 'sentence2:', sentence2])
    tokenize_result = tokenizer(inputs, padding=False, truncation=False, verbose=False, add_special_tokens=True)

    input_ids = tokenize_result.input_ids
    attention_mask = tokenize_result.attention_mask
    if is_training:
        if do_divide:
            str_gold = str(gold_probability)[0] + ' ' + str(gold_probability)[1] + ' ' + str(gold_probability)[2:]
            labels = tokenizer.encode(str_gold, add_special_tokens=False)
            labels = clean_label(labels)
        else:
            labels = tokenizer.encode(str(gold_probability), add_special_tokens=False)
    else:
        labels = None
    feature = STSFeatures(input_ids = input_ids,
                          attention_mask = attention_mask,
                          token_type_ids = None,
                          labels = labels,
                          gold_probability = gold_probability,
                          sts_id = sts_id
    )
    return feature

def STS_convert_example_to_features_cls(example, tokenizer, max_seq_length, is_training):
    sentence1 = example.sentence1
    sentence2 = example.sentence2
    gold_probability  = example.gold_probability
    sts_id = example.sts_id
    assert isinstance(tokenizer, BertTokenizer) or isinstance(tokenizer, RobertaTokenizer) , 'The tokenizer used for classification preprocessing must be member of the tokenizer of bert type.'
    if is_training:
        tokenize_result = tokenizer.encode_plus(sentence1, sentence2, max_length=max_seq_length, padding="max_length", stride=0, truncation='only_second', return_overflowing_tokens=True, return_token_type_ids=True, add_special_tokens=True)
    else:
        tokenize_result = tokenizer.encode_plus(sentence1, sentence2, padding=False, truncation=False, verbose=False)

    if is_training and tokenize_result.num_truncated_tokens > 0:
        return None
    input_ids = tokenize_result.input_ids
    attention_mask = tokenize_result.attention_mask
    if isinstance(tokenizer, RobertaTokenizer):
        token_type_ids = None
    else:
        token_type_ids = tokenize_result.token_type_ids
    feature = STSFeatures(input_ids = input_ids,
                          attention_mask = attention_mask,
                          token_type_ids = token_type_ids,
                          labels = None,
                          gold_probability = gold_probability / 5 if gold_probability is not None else None,
                          sts_id = sts_id
    )
    return feature

def STS_convert_examples_to_features(examples, tokenizer, max_seq_length, is_training, do_divide, verbose=True):
    features = []
    if is_training:
        if verbose:
            tqdm_examples = tqdm.tqdm(examples, desc='Convert STS train examples to features')
        else:
            tqdm_examples = examples 
    else:
        if verbose:
            tqdm_examples = tqdm.tqdm(examples, desc='Convert STS eva examples to features')
        else:
            tqdm_examples = examples 
    for example in tqdm_examples:
        feature = STS_convert_example_to_features(example, tokenizer, max_seq_length, is_training, do_divide)
        if feature != None:
            features.append(feature)
    # if is_training:
    #     features.sort(key=lambda feature: len(feature['input_ids']), reverse=False)
    return features

def STS_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert

def STS_convert_examples_to_features_cls(examples, tokenizer, max_seq_length, is_training, do_divide, verbose=True):
    features = []
    STS_convert_example_to_features_init(tokenizer)
    if is_training:
        if verbose:
            tqdm_examples = tqdm.tqdm(examples, desc='Convert STS train examples to features')
        else:
            tqdm_examples = examples 
    else:
        if verbose:
            tqdm_examples = tqdm.tqdm(examples, desc='Convert STS eva examples to features')
        else:
            tqdm_examples = examples 
    for example in tqdm_examples:
        feature = STS_convert_example_to_features_cls(example, tokenizer, max_seq_length, is_training)
        if feature != None:
            features.append(feature)
    # if is_training:
    #     features.sort(key=lambda feature: len(feature['input_ids']), reverse=False)
    return features


