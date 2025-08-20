import json
import tqdm
import os
import re
import random
from transformers import PreTrainedTokenizerBase, BartTokenizer

class MQA_example():
    def __init__(self, question, context, answer, qas_id):
        self.question_text = question
        self.context_text = context
        self.answer_text = answer
        self.qas_id = qas_id
        
    def print_content(self):
        print(f'question_text:\n{self.question_text}')
        print(f'context_text:\n{self.context_text}')
        print(f'answer_text:\n{self.answer_text}')
        print(f'qas_id:\n{self.qas_id}', end='\n')

class MRC_example():
    def __init__(self, question, context, options, evidences, answer, qas_id):
        self.question_text = question
        self.context_text = context
        self.options_text = options
        self.evidence_text = evidences
        self.answer_text = answer
        self.qas_id = qas_id
        
    def print_content(self):
        print(f'question_text:\n{self.question_text}')
        print(f'context_text:\n{self.context_text}')
        print(f'options_text:\n{self.options_text}')
        print(f'evidence_text:\n{self.evidence_text}')
        print(f'answer_text:\n{self.answer_text}')
        print(f'qas_id:\n{self.qas_id}', end='\n')
        
def _string_join(lst):
    inputs = ' '.join(lst)
    regex_contiguous_space = re.compile(r'\s+')
    inputs = re.sub(regex_contiguous_space, ' ', inputs)
    return inputs

def MQAFeatures(
        input_ids = None,
        attention_mask = None,
        labels = None,
        qas_id: str = None
    ):
        feature = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'qas_id': qas_id}
        return feature

def MRCFeatures(
        input_ids = None,
        attention_mask = None,
        re_labels = None,
        qa_labels = None,
        qas_id: str = None
    ):
        feature = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        're_labels': re_labels,
        'qa_labels': qa_labels,
        'qas_id': qas_id}
        return feature


def clean_num(txt):
    index = txt.find('.')
    if index != -1 and txt[:index].isnumeric():
        return txt[index+1:]
    else:
        return txt

def get_context_and_evidence(num_passage, selected, not_selected):
    sample_num = num_passage - len(selected)
    if len(not_selected) < sample_num:
        return None, None
    else:
        not_selected2 = random.sample(not_selected, k=sample_num)
    context_list = selected + not_selected2
    context_list = random.sample(context_list, k=len(context_list))
    # index_list = []
    # if len(selected) == 2:
    #     for i in selected2:
    #         index_list.append(context_list.index(i))
    #     if sorted(index_list, reverse=False) != index_list:
    #         context_list[index_list[0]], context_list[index_list[1]] = context_list[index_list[1]], context_list[index_list[0]]
    context = ' '.join(context_list)
    evidence = ' '.join(selected)
    return context, evidence

def get_MQA_RE_examples(file_path, file_type, num_passage):
    if file_type == 'eval':
        file_name = os.path.join(file_path, f'{file_type}_v2.1_public.json')
    else:
        file_name = os.path.join(file_path, f'{file_type}_v2.1.json')
    with open(file_name, 'r', encoding='utf-8') as f:
        MQA_data = json.load(f)
    RE_examples = []
    # begin_list = ['what', 'how', 'who', 'when', 'in', 'which', 'where', 'the', 'why', 'on']
    id_list = list(MQA_data['query'].keys())
    tqdm_indexes = tqdm.tqdm(range(len(id_list)), desc=f'Obtain MsMarco-RE {file_type} examples')
    id_index = 0
    for example_index in tqdm_indexes:
        example_id = id_list[example_index]
        all_passages = MQA_data['passages'][example_id]
        selected_passages = []
        not_selected_passages = []
        for passage in all_passages:
            if passage['is_selected'] == 1:
                selected_passages.append(passage['passage_text'])
            else:
                not_selected_passages.append(passage['passage_text'])
        context, evidence = get_context_and_evidence(num_passage=num_passage, 
                                                        selected=selected_passages, 
                                                        not_selected=not_selected_passages)
        if context == None:
            continue
        question = MQA_data['query'][example_id]
        qas_id = 'RE_' + str(id_index)
        id_index += 1
        # if question.split()[0].lower() in begin_list:
        RE_examples.append(MQA_example(question, context, [evidence], qas_id))
    return RE_examples
    
def get_MQA_examples(file_path, file_type):
    if file_type == 'eval':
        file_name = os.path.join(file_path, f'{file_type}_v2.1_public.json')
    else:
        file_name = os.path.join(file_path, f'{file_type}_v2.1.json')
    with open(file_name, 'r', encoding='utf-8') as f:
        MQA_data = json.load(f)
    MQA_examples = []
    begin_list = ['what', 'how', 'who', 'when', 'in', 'which', 'where', 'the', 'why', 'on']
    id_list = list(MQA_data['query'].keys())
    tqdm_indexes = tqdm.tqdm(range(len(id_list)), desc=f'Obtain MsMarco {file_type} examples')
    for example_index in tqdm_indexes:
        example_id = id_list[example_index]
        all_passages = MQA_data['passages'][example_id]
        selected_passages = []
        for passage in all_passages:
            if passage['is_selected'] == 1:
                selected_passages.append(passage['passage_text'])
        if selected_passages != []:
            context = '\n'.join(selected_passages)
        else:
            context = all_passages[0]['passage_text']
        question = MQA_data['query'][example_id]
        if MQA_data['wellFormedAnswers'][example_id] == '[]':
            answer = MQA_data['answers'][example_id]
        else:
            answer = MQA_data['wellFormedAnswers'][example_id]
        qas_id = 'MQA_' + str(example_index)
        if answer == ['No Answer Present.'] or answer == ['']:
            continue
        if question.split()[0].lower() in begin_list:
            MQA_examples.append(MQA_example(question, context, answer, qas_id))
    return MQA_examples
            

def get_MQA_MRC_examples(file_path, file_type, num_passage):
    if file_type == 'eval':
        file_name = os.path.join(file_path, f'{file_type}_v2.1_public.json')
    else:
        file_name = os.path.join(file_path, f'{file_type}_v2.1.json')
    with open(file_name, 'r', encoding='utf-8') as f:
        MQA_data = json.load(f)
    MRC_examples = []
    # begin_list = ['what', 'how', 'who', 'when', 'in', 'which', 'where', 'the', 'why', 'on']
    id_list = list(MQA_data['query'].keys())
    tqdm_indexes = tqdm.tqdm(range(len(id_list)), desc=f'Obtain MsMarco-MRC {file_type} examples')
    id_index = 0
    for example_index in tqdm_indexes:
        example_id = id_list[example_index]
        all_passages = MQA_data['passages'][example_id]
        selected_passages = []
        not_selected_passages = []
        for passage in all_passages:
            if passage['is_selected'] == 1:
                selected_passages.append(passage['passage_text'])
            else:
                not_selected_passages.append(passage['passage_text'])
        if len(selected_passages) != 1:
            continue
        context, evidence = get_context_and_evidence(num_passage=num_passage, 
                                                     selected=selected_passages, 
                                                     not_selected=not_selected_passages)
        if context == None:
            continue

        if MQA_data['wellFormedAnswers'][example_id] == '[]':
            answer = MQA_data['answers'][example_id]
        else:
            answer = MQA_data['wellFormedAnswers'][example_id]

        if answer == ['No Answer Present.'] or answer == ['']:
            continue

        question = MQA_data['query'][example_id]
        qas_id = 'RE_' + str(id_index)
        id_index += 1
        # if question.split()[0].lower() in begin_list:
        MRC_examples.append(MRC_example(question, context, None, [evidence], answer, qas_id))
    return MRC_examples

def get_ExpMRC_MRC_examples(file_name, data_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        txt = json.load(f)
    examples = []
    if 'dev' in file_name:
        file_type = 'dev'
    else:
        file_type = 'train'
    tqdm_indexes = tqdm.tqdm(range(len(txt['data'])), desc=f'Obtain {data_name}-MRC {file_type} examples')
    if data_name == 'race':
        for content_index in tqdm_indexes:
            content = txt['data'][content_index]
            context = content['article']
            for question_index in range(len(content['questions'])):
                question = clean_num(content['questions'][question_index])
                options = ' '.join([f"({chr(ord('A') + options_index)}) {content['options'][question_index][options_index]}" for options_index in range(len(content['options'][question_index]))])
                evidences = content['evidences'][question_index]
                answer = [content['options'][question_index][ord(content['answers'][question_index]) - ord('A')]]
                qas_id = content['id'] + '-' +  str(question_index)
                examples.append(MRC_example(question, context, options, evidences, answer, qas_id))
    elif data_name == 'squad':
        for index in tqdm_indexes:
            content = txt['data'][index]
            for paragraph in content['paragraphs']:
                context = paragraph['context']
                for qas in paragraph['qas']:
                    question = qas['question']
                    evidences= qas['evidences']
                    answer = list([ans['text'] for ans in qas['answers']])
                    qas_id = qas['id']
                    examples.append(MRC_example(question, context, None, evidences, answer, qas_id))
    return examples

def get_ExpMRC_Pred_MRC_examples(file_name, data_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        txt = json.load(f)
    examples = []
    if 'dev' in file_name:
        file_type = 'dev'
    else:
        file_type = 'train'
    tqdm_indexes = tqdm.tqdm(range(len(txt['data'])), desc=f'Obtain {data_name}-MRC {file_type} examples')
    if data_name == 'race':
        for content_index in tqdm_indexes:
            content = txt['data'][content_index]
            context = content['article']
            for question_index in range(len(content['questions'])):
                question = clean_num(content['questions'][question_index])
                options = ' '.join([f"({chr(ord('A') + options_index)}) {content['options'][question_index][options_index]}" for options_index in range(len(content['options'][question_index]))])
                evidences = content['evidences'][question_index]
                answer = [content['options'][question_index][ord(content['answers'][question_index]) - ord('A')]]
                qas_id = content['id'] + '-' +  str(question_index)
                examples.append(MRC_example(question, context, options, evidences, answer, qas_id))
    elif data_name == 'squad':
        for index in tqdm_indexes:
            content = txt['data'][index]
            for paragraph in content['paragraphs']:
                context = paragraph['context']
                for qas in paragraph['qas']:
                    question = qas['question']
                    evidences= None
                    answer = None
                    qas_id = qas['id']
                    examples.append(MRC_example(question, context, None, evidences, answer, qas_id))
    return examples


def get_ExpMRC_Sim_MRC_train_examples(file_name, data_name):
    with open(file_name, 'r', encoding='utf-8') as f:
        example_list = json.load(f)
    examples = []
    file_type = 'train'
    tqdm_indexes = tqdm.tqdm(range(len(example_list)), desc=f'Obtain {data_name}-MRC {file_type} examples')
    for index in tqdm_indexes:
        example = example_list[index]
        question = example['question']
        context = example['context']
        evidences= example['evidences']
        answer = example['answers']
        qas_id = example['qas_id']
        examples.append(MRC_example(question, context, None, evidences, answer, qas_id))
    return examples

def MQA_convert_example_to_features(example, tokenizer, max_seq_length, is_training):
    question = example.question_text
    context  = example.context_text
    if is_training:
        answer = example.evidence_text[0]  # encoder输入为lower，decoder输入也要为lower。
    inputs = _string_join(['question:', question, 'context:', context])
    if is_training:
        tokenize_result = tokenizer(inputs, max_length=max_seq_length, padding=False, truncation=True, return_overflowing_tokens=True, add_special_tokens=True)
    else:
        tokenize_result = tokenizer(inputs, padding=False, truncation=False, verbose=False)
    if is_training and tokenize_result.num_truncated_tokens > 0:
        return None
    input_ids = tokenize_result.input_ids
    attention_mask = tokenize_result.attention_mask
    if is_training:
        labels = tokenizer.encode(answer, add_special_tokens=False)
    else:
        labels = None
    feature = MQAFeatures(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          labels=labels, 
                          qas_id=example.qas_id)
    return feature

def convert_example_to_mrc_features(example, tokenizer, max_seq_length, is_training):
    question = example.question_text
    context  = example.context_text
    options  = example.options_text
    if is_training:
        evidence = example.evidence_text[0]  # encoder输入为lower，decoder输入也要为lower。
        answer = example.answer_text[0]  # encoder输入为lower，decoder输入也要为lower。
    if options == None:
        inputs = _string_join(['question:', question, 'context:', context])
    else:
        inputs = _string_join(['question:', question, 'options:', options, 'context:', context])
    if is_training:
        tokenize_result = tokenizer(inputs, max_length=max_seq_length, padding=False, truncation=True, return_overflowing_tokens=True, add_special_tokens=True)
    else:
        tokenize_result = tokenizer(inputs, padding=False, truncation=False, verbose=False)
    if is_training and tokenize_result.num_truncated_tokens > 0:
        return None
    input_ids = tokenize_result.input_ids
    attention_mask = tokenize_result.attention_mask
    if is_training:
        if isinstance(tokenizer, BartTokenizer):
            re_labels = tokenizer.encode(evidence, add_special_tokens=True)
            qa_labels = tokenizer.encode(answer, add_special_tokens=True)
        else:
            re_labels = tokenizer.encode(evidence, add_special_tokens=False)
            qa_labels = tokenizer.encode(answer, add_special_tokens=False)
    else:
        re_labels = None
        qa_labels = None
    feature = MRCFeatures(input_ids=input_ids, 
                          attention_mask=attention_mask, 
                          re_labels=re_labels, 
                          qa_labels=qa_labels, 
                          qas_id=example.qas_id)
    return feature


def MQA_convert_example_to_features_init(tokenizer_for_convert: PreTrainedTokenizerBase):
    global tokenizer
    tokenizer = tokenizer_for_convert

def MQA_convert_examples_to_features(examples, tokenizer, max_seq_length, is_training):
    features = []
    MQA_convert_example_to_features_init(tokenizer)
    if is_training:
        tqdm_examples = tqdm.tqdm(examples, desc='Convert MQA train examples to features')
    else:
        tqdm_examples = tqdm.tqdm(examples, desc='Convert MQA eva examples to features')
    for example in tqdm_examples:
        feature = MQA_convert_example_to_features(example, tokenizer, max_seq_length, is_training)
        if feature != None:
            features.append(feature)
    if is_training:
        features.sort(key=lambda feature: len(feature['input_ids']), reverse=False)
    return features

def convert_examples_to_mrc_features(examples, tokenizer, max_seq_length, is_training):
    features = []
    MQA_convert_example_to_features_init(tokenizer)
    if is_training:
        tqdm_examples = tqdm.tqdm(examples, desc='Convert train examples to MRC features')
    else:
        tqdm_examples = tqdm.tqdm(examples, desc='Convert eva examples to MRC features')
    for example in tqdm_examples:
        feature = convert_example_to_mrc_features(example, tokenizer, max_seq_length, is_training)
        if feature != None:
            features.append(feature)
    if is_training:
        features.sort(key=lambda feature: len(feature['input_ids']), reverse=False)
    return features

