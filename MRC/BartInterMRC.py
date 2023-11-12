#!/root/anaconda3/envs/guan/bin python

# coding: utf-8

# In[59]:
import argparse
import sys
sys.path.append('..')
import torch
import torch.nn as nn
import random
import os
import tqdm
import warnings
import json
warnings.filterwarnings('ignore')

import transformers
from BartModel import BartForConditionalGeneration, BartForMRC
from transformers import BartTokenizer, BartConfig
from MQA_preprocess import (
    get_ExpMRC_MRC_examples,
    get_ExpMRC_Pred_MRC_examples,
    get_ExpMRC_Sim_MRC_train_examples,
    convert_examples_to_mrc_features
)
from Auxiliary_Functions import get_time_str, count_parameters, set_seed
from Preprocess import (
    mycollate, 
    mycollate_mrc, 
    mycollate_eva,
    data_padding_input_ids,
    data_padding_attention_mask
)

from eval_expmrc import evaluate_span

# torch.set_printoptions(precision=10, sci_mode=False)

def load_data(tokenizer, cached_features_file, args):
    if os.path.exists(cached_features_file):
        if args.is_new:
            train_file_name = os.path.join(args.data_path, f'train-{args.mode}-squad.json')
            train_examples = get_ExpMRC_Sim_MRC_train_examples(train_file_name, args.data_name)
        else:
            train_file_name = os.path.join(args.data_path, 'train-pseudo-squad.json')
            train_examples = get_ExpMRC_MRC_examples(train_file_name, args.data_name)
        dev_file_name = os.path.join(args.data_path, 'expmrc-squad-dev.json')
        eva_examples = get_ExpMRC_MRC_examples(dev_file_name, args.data_name)

        with open(cached_features_file, 'r', encoding='utf-8') as f:
            features = json.load(f)
        train_features = features['train_features']
        eva_features = features['eva_features']
    else:
        if args.is_new:
            train_file_name = os.path.join(args.data_path, f'train-{args.mode}-squad.json')
            train_examples = get_ExpMRC_Sim_MRC_train_examples(train_file_name, args.data_name)
        else:
            train_file_name = os.path.join(args.data_path, 'train-pseudo-squad.json')
            train_examples = get_ExpMRC_MRC_examples(train_file_name, args.data_name)
        dev_file_name = os.path.join(args.data_path, 'expmrc-squad-dev.json')
        eva_examples = get_ExpMRC_MRC_examples(dev_file_name, args.data_name)
        train_features = convert_examples_to_mrc_features(examples=train_examples, 
                                                          tokenizer=tokenizer, 
                                                          max_seq_length=args.max_len, 
                                                          is_training=True)
        eva_features = convert_examples_to_mrc_features(examples=eva_examples, 
                                                        tokenizer=tokenizer,
                                                        max_seq_length=None,
                                                        is_training=False)

        features = {'train_features': train_features, 'eva_features': eva_features}
        with open(cached_features_file, 'w', encoding='utf-8') as f:
            json.dump(features, f)
    return train_features, eva_features, train_examples, eva_examples

def train(model, bart_tokenizer, cached_features_file, device, args):
    train_features, eva_features, train_examples, eva_examples = load_data(bart_tokenizer, cached_features_file, args)
    print(f'train_features_len:{len(train_features)}, eva_features_len:{len(eva_features)}')
    train_iterator = torch.utils.data.DataLoader(train_features, 
                                                 collate_fn=mycollate_mrc, 
                                                 batch_size=args.mini_batch_size,
                                                 # num_workers=8,
                                                 # pin_memory=True,
                                                 shuffle=True)
    eva_iterator = torch.utils.data.DataLoader(eva_features, 
                                               collate_fn=mycollate_eva, 
                                               batch_size=args.mini_batch_size*5,
                                               # num_workers=8,
                                               # pin_memory=True,
                                               shuffle=False)

    accumulate_step = args.batch_size / args.mini_batch_size
    num_training_steps = len(train_iterator) // accumulate_step * args.N_EPOCHS
    num_warmup_steps = int(num_training_steps * args.warm_up_rate)
    print(f'total_steps: {num_training_steps}, warmup_steps: {num_warmup_steps}')
    no_decay = ['layer_norm.weight', 'bias']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 
         'weight_decay': args.weight_decay,
        }, 
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0,
        }
    ]
    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, 
                                                             num_warmup_steps=num_warmup_steps, 
                                                             num_training_steps=num_training_steps)
    epoch_loss = 0
    global_step = 0
    best_score = 0
    for e in range(args.N_EPOCHS):
        epoch_iterator = tqdm.tqdm(train_iterator, desc='Epoch[%i]' % e)
        for step, batch in enumerate(epoch_iterator, 1):
            model.train()
            batch = tuple(item.to(device) for item in batch.values())
            inputs = {
                'input_ids':                batch[0],
                'attention_mask':           batch[1],
                're_decoder_input_ids':     batch[2],
                're_decoder_attention_mask':batch[3],
                're_labels':                batch[4],
                'qa_decoder_input_ids':     batch[5],
                'qa_decoder_attention_mask':batch[6],
                'qa_labels':                batch[7]
            }
            outputs = model(**inputs)
            loss = outputs.loss
            epoch_loss += loss.item()
            loss = loss / accumulate_step
            loss.backward()

            if step % accumulate_step == 0:
                nn.utils.clip_grad_norm_(optimizer_grouped_parameters[0]['params'], args.clip_threshold)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                if global_step % args.save_steps == 0:
                    best_score = ExpMRC_recording(model, bart_tokenizer, device, eva_iterator, eva_examples, eva_features, best_score, global_step, args)
                epoch_iterator.set_postfix(loss = '%.16f' % (epoch_loss / (step + e * len(train_iterator))), lr='%.8f' % optimizer.state_dict()['param_groups'][0]['lr'])
    return None

def ExpMRC_recording(model, bart_tokenizer, device, eva_iterator, eva_examples, eva_features, best_f1, global_step, args):
    results = inference(model, bart_tokenizer, device, eva_iterator, eva_examples, eva_features, args)
    all_f1 = results['ALL_F1']
    ans_f1 = results['ANS_F1']
    evi_f1 = results['EVI_F1']
    if all_f1 > best_f1:
        model_state_dict = {k: v.cpu() for (k, v) in model.state_dict().items()}
        torch.save(model_state_dict, args.state_file)
        # save_result()
        best_f1 = all_f1
    with open(args.log_file_name, 'a', encoding='utf-8') as f:
        f.write('ans_f1:{:.3f}, evi_f1:{:.3f}, all_f1:{:.3f}, global_step:{}\n'.format(ans_f1, evi_f1, all_f1, global_step))
    return best_f1


def BartSubModel(shared, encoder, decoder, lm_head, config):
    sub_model = BartForConditionalGeneration(config)
    sub_model.model.shared.load_state_dict(shared.state_dict())
    sub_model.model.encoder.load_state_dict(encoder.state_dict())
    sub_model.model.decoder.load_state_dict(decoder.state_dict())
    sub_model.lm_head.load_state_dict(lm_head.state_dict())
    # sub_model.lm_head.weight.data = sub_model.model.shared.weight

    return sub_model


def get_index(lst, sub_list):
    lst_len = len(lst)
    sub_len = len(sub_list)
    for i in range(lst_len - sub_len + 1):
        if lst[i:(i+sub_len)] == sub_list:
            return i
    raise ValueError('Can not find sub_list in list!')

def get_qa_inputs(re_outputs, re_input_ids, tokenizer):
    qa_input_list = []
    qa_attn_list = []
    qa_input_len_list = []
    qa_attn_len_list = []
    context_id = torch.tensor(tokenizer.encode(' context:', add_special_tokens=False), dtype=torch.long, device=re_input_ids.device)

    for i in range(re_input_ids.shape[0]):
        context_index = get_index(re_input_ids[i].tolist(), context_id.tolist())
        q_ids = re_input_ids[i][:context_index]
        # re_outputs[i]: [0, xxx, xxx, ...]
        if 1 in re_outputs[i]:
            c_ids = re_outputs[i][2:re_outputs[i].tolist().index(1)].to(re_input_ids.device)
        else:
            c_ids = re_outputs[i][2:].to(re_input_ids.device)
        qa_input = torch.cat((q_ids, context_id, c_ids), 0)
        qa_input_list.append(qa_input)
        qa_input_len_list.append(qa_input.shape[0])
        # print(f're_input_ids: {re_input_ids[i]}')
        # print(f're_output: {re_outputs[i]}')
        # print(f'q_ids: {q_ids}')
        # print(f'c_ids: {c_ids}')
        # print(f'qa_input: {qa_input}')
        # print('==='*10)

    max_len = max(qa_input_len_list)
    for i in range(re_input_ids.shape[0]):
        input_len = qa_input_list[i].shape[0]
        qa_input_list[i] = data_padding_input_ids(max_len, qa_input_list[i]) 
        qa_attn_list.append(data_padding_attention_mask(max_len, input_len, qa_input_list[i].device)) 
    input_ids = torch.stack(qa_input_list, 0)    # batch_size, qa_len, hid_dim
    attention_mask = torch.stack(qa_attn_list, 0)     # batch_size, qa_len
    return input_ids, attention_mask


def inference(model, bart_tokenizer, device, eva_iterator, eva_examples, eva_features, args):
    model.eval()
    re_model = BartSubModel(model.shared, model.encoder, model.re_decoder, model.lm_head, model.config)
    qa_model = BartSubModel(model.shared, model.encoder, model.qa_decoder, model.lm_head, model.config)

    re_model.to(device)
    qa_model.to(device)

    re_model.eval()
    qa_model.eval()
    num_beams = 1
    max_re_output_length = 200
    max_qa_output_length = 50
    predictions = {}
    epoch_iterator = tqdm.tqdm(eva_iterator, desc='Evaluate')
    batch_size = eva_iterator.batch_size
    for iterator_index, batch in enumerate(epoch_iterator):
        batch = tuple(item.to(device) for item in batch.values())
        re_outputs = re_model.generate(input_ids=batch[0],
                                       attention_mask=batch[1],
                                       # num_beams=num_beams,
                                       min_length=1,
                                       max_length=max_re_output_length,
                                       # early_stopping=True,
        )
        qa_input_ids, qa_attention_mask = get_qa_inputs(re_outputs, batch[0], bart_tokenizer)
        qa_outputs = qa_model.generate(input_ids=qa_input_ids,
                                           attention_mask=qa_attention_mask,
                                           # num_beams=num_beams,
                                           min_length=1,
                                           max_length=max_qa_output_length,
                                           # early_stopping=True,
        )
        for i in range(batch[0].shape[0]):
            # print(f'{i + iterator_index * batch_size + 1}--{torch.max(torch.nn.functional.softmax(outputs.scores[0][i]))}')
            re_pred = re_outputs[i].cpu()
            qa_pred = qa_outputs[i].cpu()
            re_pred = bart_tokenizer.decode(re_pred, skip_special_tokens=True)
            qa_pred = bart_tokenizer.decode(qa_pred, skip_special_tokens=True)
            qas_id = eva_features[i + iterator_index * batch_size]['qas_id']
            if predictions.get(qas_id) == None:
                predictions[qas_id] = {}
            predictions[qas_id]['evidence'] = re_pred
            predictions[qas_id]['answer']   = qa_pred

    if args.prediction_file == None:
        return predictions   
    with open(args.prediction_file, 'w', encoding='utf-8') as f:
        json.dump(predictions, f) 
    if args.is_training: 
        dev_file_name = os.path.join(args.data_path, 'expmrc-squad-dev.json')
        ground_truth_file= json.load(open(dev_file_name, 'rb'))
        prediction_file  = json.load(open(args.prediction_file, 'rb'))

        all_f1_score, answer_f1_score, evidence_f1_score, total_count, skip_count = evaluate_span(ground_truth_file, prediction_file)
        results = {}
        results['ALL_F1'] = all_f1_score
        results['ANS_F1'] = answer_f1_score
        results['EVI_F1'] = evidence_f1_score
        results['TOTAL'] = total_count
        return results
    else:
        return None

def test(model, tokenizer, device, args):
    model.load_state_dict(torch.load(args.state_file))

    test_examples = get_ExpMRC_Pred_MRC_examples(args.dev_file_name, args.data_name)

    test_features = convert_examples_to_mrc_features(examples=test_examples, 
                                                     tokenizer=tokenizer,
                                                     max_seq_length=None,
                                                     is_training=False)

    test_iterator = torch.utils.data.DataLoader(test_features, 
                                                collate_fn=mycollate_eva, 
                                                batch_size=args.batch_size,
                                                # num_workers=8,
                                                # pin_memory=True,
                                                shuffle=False)

    results = inference(model, tokenizer, device, test_iterator, test_examples, test_features, args)
    # all_f1 = results['ALL_F1']
    # ans_f1 = results['ANS_F1']
    # evi_f1 = results['EVI_F1']
    # total = results['TOTAL']
    # print(f'ans_f1: {ans_f1:.3f}, evi_f1: {evi_f1:.3f}, all_f1: {all_f1:.3f}, total: {total}')


def get_tokenizer(model_level):
    try:
        tokenizer = BartTokenizer.from_pretrained(model_level)
    except:
        print('Retry to get the tokenizer!')
        tokenizer = get_tokenizer(model_level)
    return tokenizer

def load_state_dict(bart_model, mrc_model):
    mrc_model.shared.load_state_dict(bart_model.model.shared.state_dict()) 
    mrc_model.encoder.load_state_dict(bart_model.model.encoder.state_dict()) 
    mrc_model.re_decoder.load_state_dict(bart_model.model.decoder.state_dict()) 
    mrc_model.qa_decoder.load_state_dict(bart_model.model.decoder.state_dict()) 
    mrc_model.lm_head.weight = mrc_model.shared.weight 
    return mrc_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=52)
    parser.add_argument('--model_name', type=str, default='Bart-InterMRC')
    parser.add_argument('--model_level', type=str, default='facebook/bart-base')
    parser.add_argument('--left_threshold', type=float, default=0.01)
    parser.add_argument('--right_threshold', type=float, default=0.3)
    parser.add_argument('--N_EPOCHS', type=int, default=10)
    parser.add_argument('--max_len', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=200)
    parser.add_argument('--mini_batch_size', type=int, default=10)
    parser.add_argument('--warm_up_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.0)
    parser.add_argument('--clip_threshold', type=float, default=1.0)
    parser.add_argument('--save_steps', type=int, default=200)
    parser.add_argument('--data_name', type=str, default='squad')
    parser.add_argument('--device', type=str, default='cuda:1')
    parser.add_argument('--mode', type=str, default='t5-base')
    parser.add_argument('--is_new', action="store_true")
    parser.add_argument('--is_training', action="store_true")
    parser.add_argument('--num', type=int, default=1)
    parser.add_argument('--data_path', type=str, default='./Data/Datasets/ExpMRC/')
    
    args = parser.parse_args()

    log_file_default_values = {'log_file_name': f"./loss_{args.model_level.split('/')[-1]}{args.num}.txt"}
    state_file_default_values = {'state_file': f"./Data/{args.model_level.split('/')[-1]}{get_time_str()}{args.num}.pt"}
    prediction_file_default_values = {'prediction_file': f"./pred_{args.model_level.split('/')[-1]}{args.num}.json"}
    parser.set_defaults(**log_file_default_values)
    parser.set_defaults(**state_file_default_values)
    parser.set_defaults(**prediction_file_default_values)
    args = parser.parse_args()

    bart_model = BartForConditionalGeneration.from_pretrained(args.model_level)
    set_seed(args.random_seed)
    bart_tokenizer = get_tokenizer(args.model_level)
    config = BartConfig.from_pretrained(args.model_level)
    config.tokenizer = bart_tokenizer
    config.f1_threshold = None
    config.left_threshold = args.left_threshold
    config.right_threshold = args.right_threshold
    # config.decoder_start_token_id = 0
    model = BartForMRC(config)
    model = load_state_dict(bart_model, model)
    assert args.batch_size % args.mini_batch_size == 0
    device = torch.device(args.device)
    model = model.to(device)
    with open(args.log_file_name, 'a', encoding='utf-8') as f:
        f.write(f'The MRC model has {count_parameters(model):,} parameters. \n')
    cached_features_file = os.path.join('./Data/', 'padding_bart_{}_{}_{}.json'.format(args.mode, args.data_name, str(args.max_len)))
    train(model, bart_tokenizer, cached_features_file, device, args)
    
