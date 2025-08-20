#!/root/anaconda3/envs/guan/bin python

# coding: utf-8

# In[59]:


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
sys.path.append('../..')
import transformers
from transformers import BertPreTrainedModel, BertModel, BertConfig, BertTokenizer, AutoModelForSequenceClassification
from transformers import RobertaPreTrainedModel, RobertaModel, RobertaConfig, RobertaTokenizer
from transformers.file_utils import ModelOutput
from STS_preprocess import get_STS_examples, STS_convert_examples_to_features_cls
from scipy.stats import spearmanr, pearsonr
from Auxiliary_Functions import (
    time_since, 
    get_time_str, 
    count_parameters, 
    set_seed, 
    string_to_float,
)
import argparse

def load_data(data_name, file_path, max_len, tokenizer, cached_features_file):
    if os.path.exists(cached_features_file):
        train_examples = get_STS_examples(data_name, file_path, 'train', None, do_augment=False)
        eva_examples = get_STS_examples(data_name, file_path, 'dev', None, do_augment=False)

        with open(cached_features_file, 'r', encoding='utf-8') as f:
            features = json.load(f)
        train_features = features['train_features']
        eva_features = features['eva_features']

    else:
        train_examples = get_STS_examples(data_name, file_path, 'train', None, do_augment=False)
        eva_examples = get_STS_examples(data_name, file_path, 'dev', None, do_augment=False)

        train_features = STS_convert_examples_to_features_cls(examples=train_examples, 
                                                          tokenizer=tokenizer, 
                                                          max_seq_length=max_len, 
                                                          is_training=True,
                                                          do_divide=False)
        eva_features = STS_convert_examples_to_features_cls(examples=eva_examples, 
                                                        tokenizer=tokenizer,
                                                        max_seq_length=max_len,
                                                        is_training=False,
                                                        do_divide=False)
        features = {'train_features': train_features, 'eva_features': eva_features}
        # with open(cached_features_file, 'w', encoding='utf-8') as f:
        #     json.dump(features, f)
    return train_features, eva_features, train_examples, eva_examples


def data_padding_eva(max_len, data):
    new_data = torch.zeros(max_len, dtype=torch.long)
    data = torch.tensor(data, dtype=torch.long)
    new_data[:len(data)] = data
    return new_data

def mycollate_bert(datasets):
    new_datasets = list()
    for example in datasets:
        new_datasets.append({'input_ids': torch.tensor(example['input_ids']), 
                             'attention_mask': torch.tensor(example['attention_mask']), 
                             'token_type_ids': torch.tensor(example['token_type_ids']) if example['token_type_ids'] is not None else None, 
                             'gold_probability': torch.tensor(example['gold_probability'], dtype=torch.float32)})
    new_datasets2 = dict()
    new_datasets2['input_ids'] = torch.stack([example['input_ids'] for example in new_datasets], 0)
    new_datasets2['attention_mask'] = torch.stack([example['attention_mask'] for example in new_datasets], 0)
    new_datasets2['token_type_ids'] = torch.stack([example['token_type_ids'] for example in new_datasets], 0) if example['token_type_ids'] is not None else None
    new_datasets2['gold_probability'] = torch.stack([example['gold_probability'] for example in new_datasets], 0)
    return new_datasets2


def mycollate_eva_bert(datasets):
    len_list = [len(example['input_ids']) for example in datasets]
    max_len = max(len_list)
    new_datasets = list()
    for example in datasets:
        new_datasets.append({'input_ids': data_padding_eva(max_len, example['input_ids']), 
                             'attention_mask': data_padding_eva(max_len, example['attention_mask']), 
                             'token_type_ids': data_padding_eva(max_len, example['token_type_ids']) if example['token_type_ids'] is not None else None})
    new_datasets2 = dict()
    new_datasets2['input_ids'] = torch.stack([example['input_ids'] for example in new_datasets], 0)
    new_datasets2['attention_mask'] = torch.stack([example['attention_mask'] for example in new_datasets], 0)
    new_datasets2['token_type_ids'] = torch.stack([example['token_type_ids'] for example in new_datasets], 0) if example['token_type_ids'] is not None else None
    return new_datasets2

class ModelOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    sts_logits: torch.FloatTensor = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    all_attentions: Optional[Tuple[torch.FloatTensor]] = None  
        

class BertForSemanticTextualSimilarity(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForSemanticTextualSimilarity, self).__init__(config)
        self.num_labels = config.num_labels
        self.bert = BertModel(config)
        self.sts_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, gold_probability=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        last_hidden_state = outputs[0]
        first_word = last_hidden_state[:, 0, :]
        classify_logits = self.sts_outputs(first_word).squeeze(-1)
        classify_logits = nn.Softmax(dim=-1)(classify_logits)
        if gold_probability is not None:
            criterion = nn.MSELoss()
            loss = criterion(classify_logits[:, 0].view(-1), gold_probability.view(-1))
            return ModelOutput(
                loss=loss,
                sts_logits=classify_logits,
                all_hidden_states=outputs.hidden_states,
                all_attentions=outputs.attentions)
        return ModelOutput(
                sts_logits=classify_logits,
                all_hidden_states=outputs.hidden_states,
                all_attentions=outputs.attentions)

class RobertaForSemanticTextualSimilarity(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForSemanticTextualSimilarity, self).__init__(config)
        self.num_labels = config.num_labels
        self.roberta = RobertaModel(config)
        self.sts_outputs = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, gold_probability=None):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds
        )
        last_hidden_state = outputs[0]
        first_word = last_hidden_state[:, 0, :]
        classify_logits = self.sts_outputs(first_word).squeeze(-1)
        classify_logits = nn.Softmax(dim=-1)(classify_logits)
        if gold_probability is not None:
            criterion = nn.MSELoss()
            loss = criterion(classify_logits[:, 0].view(-1), gold_probability.view(-1))
            return ModelOutput(
                loss=loss,
                sts_logits=classify_logits,
                all_hidden_states=outputs.hidden_states,
                all_attentions=outputs.attentions)
        return ModelOutput(
                sts_logits=classify_logits,
                all_hidden_states=outputs.hidden_states,
                all_attentions=outputs.attentions)


def train(similarity_model, tokenizer, cached_features_file, device, args):
    train_features, eva_features, train_examples, eva_examples = load_data(args.dataset_name, args.data_path, args.max_len, tokenizer, cached_features_file)
    print(f'train_features_len:{len(train_features)}, eva_features_len:{len(eva_features)}')
    train_iterator = torch.utils.data.DataLoader(train_features,
                                                 collate_fn=mycollate_bert,
                                                 batch_size=args.mini_batch_size,
                                                 shuffle=True)
    eva_iterator = torch.utils.data.DataLoader(eva_features, 
                                               collate_fn=mycollate_eva_bert, 
                                               batch_size=args.mini_batch_size,
                                               shuffle=False)

    # spearman_coefficient = evaluate(similarity_model, args.device, eva_iterator, eva_examples, eva_features)[0]
    # print(f'spearman_coefficient:{spearman_coefficient:.8f}')
    accumulate_step = args.batch_size / args.mini_batch_size
    num_training_steps = len(train_iterator) // accumulate_step * args.num_epochs
    num_warmup_steps = int(num_training_steps * args.warm_up_rate)
    print(num_training_steps, num_warmup_steps)
    no_decay = ['bias', 'LayerNorm.weight']
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
    # amp.register_half_function(torch, 'einsum')
    # model, optimizer = amp.initialize(model, optimizer, opt_level='O1')
    # output_knot = math.floor(len(train_iterator) / 2 / 500) * 500
    epoch_loss = 0
    global_step = 0
    max_coefficient = -2
    for e in range(args.num_epochs):
        epoch_iterator = tqdm.tqdm(train_iterator, desc='Epoch[%i]' % e)
        for step, batch in enumerate(epoch_iterator, 1):
            similarity_model.train()
            batch = tuple(item.to(device) for item in batch.values() if item is not None)
            if len(batch) == 4:
                inputs = {
                    'input_ids':       batch[0],
                    'attention_mask':  batch[1],
                    'token_type_ids':  batch[2],
                    'gold_probability':batch[3]
                }
            else:
                inputs = {
                    'input_ids':       batch[0],
                    'attention_mask':  batch[1],
                    'gold_probability':batch[2]
                }
            outputs = similarity_model(**inputs)
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
                    spearman_coefficient, pearson_coefficient = evaluate(similarity_model, device, eva_iterator, eva_examples, eva_features)

                    coefficient = spearman_coefficient + pearson_coefficient
                    if coefficient > max_coefficient:
                        model_state_dict = {k: v.cpu() for (k, v) in similarity_model.state_dict().items()}
                        torch.save(model_state_dict, args.state_file)
                        # save_result()
                        max_coefficient = coefficient
                    with open(args.log_file_name, 'a', encoding='utf-8') as f:
                        f.write('pearson_coefficient:{:.8f}, spearman_coefficient:{:.8f}, global_step:{}\n'.format(pearson_coefficient, spearman_coefficient, global_step))
                epoch_iterator.set_postfix(loss = '%.16f' % (epoch_loss / (step + e * len(train_iterator))), lr='%.8f' % optimizer.state_dict()['param_groups'][0]['lr'])
    return args.state_file


def evaluate(similarity_model, device, eva_iterator, eva_examples, eva_features):
    similarity_model.eval()
    epoch_iterator = tqdm.tqdm(eva_iterator, desc='Evaluate')
    batch_size = eva_iterator.batch_size
    epoch_loss = 0
    pred_logits = []
    gold_logits = []
    for iterator_index, batch in enumerate(epoch_iterator):
        batch = tuple(item.to(device) for item in batch.values() if item is not None)
        if len(batch) == 3:
            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1],
                'token_type_ids':  batch[2]
            }
        else:
            inputs = {
                'input_ids':       batch[0],
                'attention_mask':  batch[1]
            }
        outputs = similarity_model(**inputs)
        pred_logits.extend((outputs.sts_logits[:, 0] * 5).tolist())
        for i in range(batch[0].shape[0]):
            index = i + iterator_index * batch_size
            gold_logits.append(eva_features[index]['gold_probability'] * 5)
        # epoch_loss += outputs.loss.item()
    return (spearmanr(pred_logits, gold_logits)[0] * 100, pearsonr(pred_logits, gold_logits)[0] * 100)


def load_test_data(data_name, file_path, tokenizer):
    test_examples = get_STS_examples(data_name, file_path, 'test', None, do_augment=False)
    test_features = STS_convert_examples_to_features_cls(examples=test_examples, 
                                                     tokenizer=tokenizer, 
                                                     max_seq_length=None, 
                                                     is_training=False,
                                                     do_divide=False)
    return test_examples, test_features


def test(similarity_model, tokenizer, device, args):
    test_examples, test_features = load_test_data(args.dataset_name, args.data_path, tokenizer)
    similarity_model.load_state_dict(torch.load(args.state_file))
    if args.log_file_name != None:
        with open(args.log_file_name, 'a', encoding='utf-8') as f:
            f.write(f'test_features_len:{len(test_features)}\n')
    test_iterator = torch.utils.data.DataLoader(test_features, 
                                                collate_fn=mycollate_eva_bert, 
                                                batch_size=args.mini_batch_size,
                                                shuffle=False)
    spearman_coefficient, pearson_coefficient = evaluate(similarity_model, device, test_iterator, test_examples, test_features)
    if args.log_file_name == None:
        print(f'pearson_coefficient:{pearson_coefficient:.8f}, spearman_coefficient:{spearman_coefficient:.8f}')
    else:
        with open(args.log_file_name, 'a', encoding='utf-8') as f:
            f.write(f'pearson_coefficient:{pearson_coefficient:.8f}, spearman_coefficient:{spearman_coefficient:.8f}\n')
    return spearman_coefficient, pearson_coefficient

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_name_or_path', type=str, default='bert-large-uncased')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--log_file_name', type=str, default='./bert_results.txt')
    parser.add_argument('--num_epochs', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--clip_threshold', type=float, default=1.0)
    parser.add_argument('--max_len', type=int, default=128, help='The max length of input_ids in model.')
    parser.add_argument('--mini_batch_size', type=int, default=40,  help='Batch size before gradient accumulation.')
    parser.add_argument('--batch_size',      type=int, default=200, help='Batch size after gradient accumulation.')
    parser.add_argument('--warm_up_rate', type=float, default=0.1, help='Ratio of warmup step in the whole training step.')
    parser.add_argument('--save_steps', type=int, default=50, help='The number of interval steps to save the model.')
    parser.add_argument('--dataset_name', type=str, default='STSb')
    parser.add_argument('--data_path', type=str, default='./../Data/Datasets/', help='The path of datasets.')
    parser.add_argument('--state_file', type=str, default='./output/BERT_States/STS_Bert_base.pt', help='The path to save the model state dict.')
    parser.add_argument('--cached_features_file_path', type=str, default='./output/', help='The cache path of training data.')
    args = parser.parse_args()
    set_seed(args.random_seed)
    device = torch.device(args.device)
    if 'roberta' in args.model_name_or_path:
        config = RobertaConfig.from_pretrained(args.model_name_or_path)
        roberta_model = RobertaModel.from_pretrained(args.model_name_or_path)
        similarity_model = RobertaForSemanticTextualSimilarity(config)
        similarity_model.roberta = roberta_model
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
    else:
        config = BertConfig.from_pretrained(args.model_name_or_path)
        bert_model = BertModel.from_pretrained(args.model_name_or_path)
        similarity_model = BertForSemanticTextualSimilarity(config)
        similarity_model.bert = bert_model
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)

    similarity_model = similarity_model.to(args.device)
    assert args.batch_size % args.mini_batch_size == 0, 'batch_size must be an integral multiple of mini_batch_size'

    with open(args.log_file_name, 'a', encoding='utf-8') as f:
        f.write(f'The STS model has {count_parameters(similarity_model):,} parameters. \n')
    cached_features_file = os.path.join(args.cached_features_file_path, '{}_no_padding_cached_{}.json'.format(args.dataset_name, str(args.max_len)))
    train(similarity_model, tokenizer, cached_features_file, device, args)
    test(similarity_model, tokenizer, device, args)
