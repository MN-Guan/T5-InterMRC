import sys
sys.path.append('..')
import argparse
from STS.Test import test
from STS.STS_preprocess import STS_example, STS_convert_examples_to_features
from MQA_preprocess import get_ExpMRC_MRC_examples
from T5Model import T5ForConditionalGeneration, T5Tokenizer
import torch
import tqdm
import collections
from eval_expmrc import evaluate_span
import re
import json
import nltk
import copy
import os
from Auxiliary_Functions import set_seed, compute_f1
from nltk import tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

def sent_tokenize(context, answers):
    masked_token = 'M_A_S_K_E_D'
    context = context.replace(answers[0], masked_token)
    sentences = nltk.sent_tokenize(context)
    for i in range(len(sentences)):
       sentences[i] = sentences[i].replace(masked_token, answers[0])
    return sentences

def get_f1_score(pred_examples, gold_examples, test_mode):
    assert len(pred_examples) == len(gold_examples), 'The length of pred_examples must be same with the length of gold_examples!'
    predictions = {}
    pred_evis = {example.qas_id: example.evidence_text[0] for example in pred_examples}
    gold_evis = {example.qas_id: example.evidence_text[0] for example in gold_examples}
    predictions = {key: {'evidence': pred_evis[key], 'answer': ''} for key in pred_evis.keys()}
    f1_list = [compute_f1(pred_evis[key], gold_evis[key]) for key in pred_evis.keys()]
    if test_mode == 'official':
        prediction_file = './test_annotation.json'
        data_path = './Data/Datasets/ExpMRC/'
        with open(prediction_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f)
        dev_file_name = os.path.join(data_path, 'expmrc-squad-dev.json')
        ground_truth_data   = json.load(open(dev_file_name, 'rb'))
        prediction_data     = json.load(open(prediction_file, 'rb'))

        all_f1_score, answer_f1_score, evidence_f1_score, total_count, skip_count = evaluate_span(ground_truth_data, prediction_data)
        return evidence_f1_score / 100
    elif test_mode == 'squad_f1':
        return sum(f1_list) / len(f1_list) 
    else:
        raise ValueError("test_mode must be one of the ['official', 'squad_f1']")

def find_evidence(context, answers):
    sentences = sent_tokenize(context, answers)
    evi_list = []
    for sentence in sentences:
        if answers[0] in sentence:
            evi_list.append(sentence)
    return evi_list

def get_new_examples(model, examples, batch_size, tokenizer, device, sts_mode=None, using_sts=True):
    new_examples = []
    multi_evi_dict = {}
    multi_evi_examples = []
    sts_eva_examples = []
    tqdm_indexes = tqdm.tqdm(range(len(examples)), desc=f'Obtain new squad-MRC train examples')
    for index in tqdm_indexes:
        context = examples[index].context_text
        answers = examples[index].answer_text
        question = examples[index].question_text
        sts_id = examples[index].qas_id
        found_evidences = find_evidence(context, answers)
        if len(found_evidences) == 0:
            print(f"Cannot find Error: {sts_id}")
        elif len(found_evidences) == 1:
            examples[index].evidence_text = found_evidences
            new_examples.append(examples[index])
        else:
            qa = ' '.join([question, answers[0]])
            for evidence in found_evidences:
                sts_eva_examples.append(STS_example(evidence, qa, None, sts_id))
            multi_evi_dict[sts_id] = found_evidences
            multi_evi_examples.append(examples[index])
    if using_sts == False:
        return new_examples
    if sts_mode == 'f1_score':
        tqdm_index = tqdm.tqdm(range(len(multi_evi_examples)), 'Evaluate')
        for index in tqdm_index:
            example = multi_evi_examples[index]
            question = example.question_text
            qas_id = example.qas_id
            answers = example.answer_text
            qa = ' '.join([question, answers[0]])
            f1_scores = [compute_f1(evi, qa) for evi in multi_evi_dict[qas_id]]
            max_f1_evi = [multi_evi_dict[qas_id][f1_scores.index(max(f1_scores))]]
            example.evidence_text = max_f1_evi
            new_examples.append(example)
        return new_examples
            
    eva_features = STS_convert_examples_to_features(examples=sts_eva_examples, 
                                                    tokenizer=tokenizer,
                                                    max_seq_length=None,
                                                    is_training=False,
                                                    do_divide=False,
                                                    verbose=True)
    model = model.to(device)
    preds = test(model, eva_features, sts_eva_examples, batch_size, device, tokenizer, verbose=True)      
    for example in multi_evi_examples:
        qas_id = example.qas_id
        pred_sts_score = preds[qas_id]
        copy_scores = copy.deepcopy(pred_sts_score)
        copy_scores.sort(reverse=True)
        # if copy_scores[0] - copy_scores[1] <= 0.5 or copy_scores[0] <= 2.5:
        #     continue
        evidence = [multi_evi_dict[qas_id][pred_sts_score.index(max(pred_sts_score))]]
        example.evidence_text = evidence
        new_examples.append(example)
    return new_examples

def convert_object_to_dict(example):
    example_dict = {}
    example_dict['question'] = example.question_text
    example_dict['context'] = example.context_text
    example_dict['evidences'] = example.evidence_text
    example_dict['answers'] = example.answer_text
    example_dict['qas_id'] = example.qas_id
    return example_dict


def write_data(examples, cached_file_name):
    example_list = []
    for example in examples:
        example_dict = convert_object_to_dict(example)
        example_list.append(example_dict)
    with open(cached_file_name, 'w', encoding='utf-8') as f:
        json.dump(example_list, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=52)
    parser.add_argument('--model_name', type=str, default='t5-base')
    parser.add_argument('--is_test', action="store_true", help="Whether to test the precision in the dev set")
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--state_file', type=str, default='./../STS/Data/Model_States/STS_T5.pt')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--sts_mode', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.random_seed)
    if args.is_test:
        examples = get_ExpMRC_MRC_examples('./Data/Datasets/ExpMRC/expmrc-squad-dev.json', 'squad')
    else:
        examples = get_ExpMRC_MRC_examples('./Data/Datasets/ExpMRC/train-pseudo-squad.json', 'squad')

    tokenizer = T5Tokenizer.from_pretrained(args.model_name)
    model = T5ForConditionalGeneration.from_pretrained(args.model_name)
    device = torch.device(args.device)
    model.load_state_dict(torch.load(args.state_file))
    new_examples = get_new_examples(model, examples, args.batch_size, tokenizer, device, args.sts_mode, using_sts=True)
    print(f'Original dataset size: {len(examples)}')
    print(f'New dataset size: {len(new_examples)}')
    if args.is_test:
        test_mode = 'official'
        f1_score = get_f1_score(new_examples, examples, test_mode)
        print(f'test_f1_score: {f1_score * 100:.2f}')
    else:
        cached_file_name = f'./Data/Datasets/ExpMRC/train-{args.model_name}-squad.json'
        write_data(new_examples, cached_file_name)

