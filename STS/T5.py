#!/root/anaconda3/envs/guan/bin python

# coding: utf-8

# In[59]:

import argparse
import torch
import torch.nn as nn
import random
import os
import tqdm
import time
import numpy as np
import warnings
import math
import json
from typing import Optional, Tuple
warnings.filterwarnings('ignore')
import sys
sys.path.append('..')
import transformers
from transformers.file_utils import ModelOutput
from transformers import T5Tokenizer
from T5Model import T5ForConditionalGeneration
from STS_preprocess import (
    get_STS_examples, 
    STS_convert_examples_to_features
)
from scipy.stats import spearmanr, pearsonr
from Auxiliary_Functions import (
    time_since, 
    get_time_str, 
    count_parameters, 
    set_seed, 
    string_to_float,
)
from Preprocess import (
    mycollate, 
    mycollate_eva,
)



def load_data(tokenizer, cached_features_file, args):
    if os.path.exists(cached_features_file):
        train_examples = get_STS_examples(args.dataset_name, args.data_path, 'train', args.train_interval, do_augment=args.do_augment)
        eva_examples = get_STS_examples(args.dataset_name, args.data_path, 'dev', None, do_augment=False)

        with open(cached_features_file, 'r', encoding='utf-8') as f:
            features = json.load(f)
        train_features = features['train_features']
        eva_features = features['eva_features']

    else:
        train_examples = get_STS_examples(args.dataset_name, args.data_path, 'train', args.train_interval, do_augment=args.do_augment)
        eva_examples = get_STS_examples(args.dataset_name, args.data_path, 'dev', None, do_augment=False)
        train_features = STS_convert_examples_to_features(examples=train_examples, 
                                                          tokenizer=tokenizer, 
                                                          max_seq_length=args.max_len, 
                                                          is_training=True,
                                                          do_divide=args.do_divide)
        eva_features = STS_convert_examples_to_features(examples=eva_examples, 
                                                        tokenizer=tokenizer,
                                                        max_seq_length=args.max_len,
                                                        is_training=False,
                                                        do_divide=False)
        features = {'train_features': train_features, 'eva_features': eva_features}
        with open(cached_features_file, 'w', encoding='utf-8') as f:
            json.dump(features, f)
    return train_features, eva_features, train_examples, eva_examples


def train(similarity_model, tokenizer, cached_features_file, device, args):
    train_features, eva_features, train_examples, eva_examples = load_data(tokenizer, cached_features_file, args)
    print(f'train_features_len:{len(train_features)}, eva_features_len:{len(eva_features)}')
    train_iterator = torch.utils.data.DataLoader(train_features,
                                                 collate_fn=mycollate,
                                                 batch_size=args.mini_batch_size,
                                                 shuffle=True)
    eva_iterator = torch.utils.data.DataLoader(eva_features, 
                                               collate_fn=mycollate_eva, 
                                               batch_size=args.mini_batch_size,
                                               shuffle=False)
    accumulate_step = args.batch_size / args.mini_batch_size
    num_training_steps = len(train_iterator) // accumulate_step * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warm_up_rate)
    print('total_steps:', num_training_steps, 'warmup_steps:', num_warmup_steps)
    no_decay = ['layer_norm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in similarity_model.named_parameters() if not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay,
        }, 
        {'params': [p for n, p in similarity_model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
        }
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                             num_warmup_steps=num_warmup_steps, 
                                                             num_training_steps=num_training_steps)
    epoch_loss = 0
    global_step = 0
    max_coefficient = -200
    for e in range(args.num_epochs):
        epoch_iterator = tqdm.tqdm(train_iterator, desc='Epoch[%i]' % e)
        for step, batch in enumerate(epoch_iterator, 1):
            similarity_model.train()
            batch = tuple(item.to(device) for item in batch.values())
            inputs = {
                'input_ids':      batch[0],
                'attention_mask': batch[1],
                'labels':         batch[4]
            }
            outputs = similarity_model(**inputs)
            # criterion = nn.MSELoss()
            # loss = criterion(outputs.sts_logits[:, 0].view(-1), batch[2].view(-1))
            loss = outputs.loss
            epoch_loss += loss.item()
            loss = loss / accumulate_step
            # with amp.scale_loss(loss, optimizer) as scaled_loss:
            #     scaled_loss.backward()
            loss.backward()
            if step % accumulate_step == 0:
                # nn.utils.clip_grad_norm_(filter(lambda item: item.requires_grad, model.parameters()), clip_threshold)
                nn.utils.clip_grad_norm_(optimizer_grouped_parameters[0]['params'], args.clip_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.save_steps == 0:
                    spearman_coefficient, pearson_coefficient = evaluate(similarity_model, tokenizer, device, eva_iterator, eva_examples, eva_features)
                    coefficient = spearman_coefficient + pearson_coefficient
                    if coefficient > max_coefficient:
                        model_state_dict = {k: v.cpu() for (k, v) in similarity_model.state_dict().items()}
                        torch.save(model_state_dict, args.state_file)
                        max_coefficient = coefficient
                    with open(args.log_file_name, 'a', encoding='utf-8') as f:
                        f.write('pearson_coefficient:{:.8f}, spearman_coefficient:{:.8f}, global_step:{}\n'.format(pearson_coefficient, spearman_coefficient, global_step))
                epoch_iterator.set_postfix(loss = '%.16f' % (epoch_loss / (step + e * len(train_iterator))), lr='%.8f' % optimizer.state_dict()['param_groups'][0]['lr'])
    


def evaluate(similarity_model, tokenizer, device, eva_iterator, eva_examples, eva_features):
    similarity_model.eval()
    num_beams = 1
    max_output_length = 10
    predictions = []
    golds = []
    epoch_iterator = tqdm.tqdm(eva_iterator, desc='Evaluate')
    batch_size = eva_iterator.batch_size
    for iterator_index, batch in enumerate(epoch_iterator):
        batch = tuple(item.to(device) for item in batch.values())
        outputs = similarity_model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=num_beams,
                                 min_length=1,
                                 max_length=max_output_length,
        )
        for i in range(batch[0].shape[0]):
            pred = tokenizer.decode(outputs[i], skip_special_tokens=True)
            pred = pred.replace(' ', '')
            prediction = string_to_float(pred)
            predictions.append(prediction)
            golds.append(eva_features[i + iterator_index * batch_size]['gold_probability'])
    return (spearmanr(predictions, golds)[0] * 100, pearsonr(predictions, golds)[0] * 100)

def load_test_data(tokenizer, args):
    test_examples = get_STS_examples(args.dataset_name, args.data_path, 'test', None, do_augment=False)
    test_features = STS_convert_examples_to_features(examples=test_examples, 
                                                     tokenizer=tokenizer, 
                                                     max_seq_length=None, 
                                                     is_training=False,
                                                     do_divide=False)
    return test_examples, test_features


def test(similarity_model, tokenizer, device, args):
    test_examples, test_features = load_test_data(tokenizer, args)
    similarity_model.load_state_dict(torch.load(args.state_file))
    if args.log_file_name != None:
        with open(args.log_file_name, 'a', encoding='utf-8') as f:
            f.write(f'test_features_len:{len(test_features)}\n')
    test_iterator = torch.utils.data.DataLoader(test_features, 
                                                collate_fn=mycollate_eva, 
                                                batch_size=args.batch_size,
                                                shuffle=False)
    spearman_coefficient, pearson_coefficient = evaluate(similarity_model, tokenizer, device, test_iterator, test_examples, test_features)
    if args.log_file_name == None:
        print(f'pearson_coefficient:{pearson_coefficient:.8f}, spearman_coefficient:{spearman_coefficient:.8f}')
    else:
        with open(args.log_file_name, 'a', encoding='utf-8') as f:
            f.write(f'pearson_coefficient:{pearson_coefficient:.8f}, spearman_coefficient:{spearman_coefficient:.8f}\n')
    return spearman_coefficient, pearson_coefficient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_file_name', type=str, default='./loss.txt')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1.5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_threshold', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=128, help='The max length of input_ids in model.')
    parser.add_argument('--mini_batch_size', type=int, default=50,  help='Batch size before gradient accumulation.')
    parser.add_argument('--batch_size',      type=int, default=200, help='Batch size after gradient accumulation.')
    parser.add_argument('--warm_up_rate', type=float, default=0.1, help='Ratio of warmup step in the whole training step.')
    parser.add_argument('--save_steps', type=int, default=50, help='The number of interval steps to save the model.')
    parser.add_argument('--train_interval', type=float, default=0.1, help='The interval of adjacent labels in train dataset.')
    parser.add_argument('--dataset_name', type=str, default='STSb')
    parser.add_argument('--do_augment', action="store_false", help='Whether to expand the train dataset?')
    parser.add_argument('--do_divide', action="store_false", help='Whether to divide the gold label of train dataset?')
    parser.add_argument('--data_path', type=str, default='./Data/Datasets/', help='The path of datasets.')
    parser.add_argument('--state_file', type=str, default='./Data/Model_States/STS_T5.pt', help='The path to save the model state dict.')
    parser.add_argument('--cached_features_file_path', type=str, default='./Data/', help='The cache path of training data.')
    args = parser.parse_args()
    set_seed(args.random_seed)
    device = torch.device(args.device)
    similarity_model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    similarity_model = similarity_model.to(args.device)
    assert args.batch_size % args.mini_batch_size == 0, 'batch_size must be an integral multiple of mini_batch_size'
    similarity_model.config.train_interval = args.train_interval
    with open(args.log_file_name, 'a', encoding='utf-8') as f:
        f.write(f'The STS model has {count_parameters(similarity_model):,} parameters. \n')
    cached_features_file = os.path.join(args.cached_features_file_path, '{}_no_padding_cached_{}_{}{}{}.json'.format(args.dataset_name, str(args.train_interval), str(args.max_len), '_aug' if args.do_augment else '', '_div' if args.do_divide else ''))
    train(similarity_model, tokenizer, cached_features_file, device, args)
    test(similarity_model, tokenizer, device, args)
