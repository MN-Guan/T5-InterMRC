import collections
import re
import rouge
import copy
import tqdm
import os
from transformers.data.metrics.squad_metrics import compute_exact, compute_f1, normalize_answer


def score_string_similarity_cover(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0

def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")

# Temporary fix for bug where {}^<\` characters roundtrip into \u2047 (??) character
def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)

def get_squad_scores(examples, preds, tokenizer, result_file_name):
    """
    Computes the exact and f1 scores from the examples and the model predictions
    """
    if os.path.exists(result_file_name):
        os.remove(result_file_name)
    exact_scores = {}
    f1_scores = {}

    for example in examples:
        qas_id = example.qas_id
        gold_answers = [answer["text"] for answer in example.answers if normalize_answer(answer["text"])]

        if not gold_answers:
            # For unanswerable questions, only correct answer is empty string
            gold_answers = [""]

        if qas_id not in preds:
            print(f"Missing prediction for {qas_id}")
            continue

        prediction = tokenizer.decode(preds[qas_id], skip_special_tokens=True)
        exact_scores[qas_id] = max(compute_exact(a, prediction) for a in gold_answers)
        f1_scores[qas_id] = max(compute_f1(a, prediction) for a in gold_answers)

        if exact_scores[qas_id] != 1:
            with open(result_file_name, 'a', encoding='utf-8') as f:
                for i in gold_answers:
                    f.write(f'{i}\n')
                f.write(f'\t\t----{prediction}\n')

    return exact_scores, f1_scores

def find_example(examples, qas_id):
    for example in examples:
        if example.qas_id == qas_id:
            return example
    raise KeyError(f'can not find {qas_id} in examples')

def get_rouge_scores(eva_examples, predictions):
    assert len(eva_examples) == len(predictions), "please check the lengths of 'eva_examples' and 'predictions'"
    gold_answers = {}
    rouge_scores = {}
    for example in eva_examples:
        qas_id = example.qas_id
        gold_answers[qas_id] = example.answer_text
    assert len(gold_answers) == len(predictions), "please check the lengths of 'gold_answers' and 'predictions'"
    tqdm_ids = tqdm.tqdm(predictions.keys(), desc='Calculate Rough-L')
    num = 1
    for qas_id in tqdm_ids:
        pred = predictions[qas_id]['answer']
        rouge_scores[qas_id] = rouge_score(pred, gold_answers[qas_id])
        # with open('./Code/T5/MsMarco/Data/False.txt', 'a', encoding='utf-8') as f:
        #     example = find_example(eva_examples, qas_id)
        #     f.write(f'\n----------{num}---------------\n')
        #     f.write(f'Rouge_score:\n{rouge_scores[qas_id]}\n') 
        #     f.write(f"Context:\n{example.context_text}\n")
        #     f.write(f"Question:\n{example.question_text}\n")
        #     f.write(f"Evidences:\n{example.answer_text}\n")
        #     f.write(f'Pred:\n{pred}\n')
        num += 1
    total = len(rouge_scores)
    result = collections.OrderedDict(
        [
            ('rouge_l', 100 * sum(rouge_scores.values()) / total),
            ('total', total)
        ]
    )
    return result

def rouge_score(prediction, ground_truths):
    rouge_l = rouge.Rouge(
        metrics=["rouge-l"]
    )
    scores = []
    for ground_truth in ground_truths:
        # if ground_truth in prediction:
        #     scores.append(1)
        # else:
        #     scores.append(0)
        if (prediction == '' and ground_truth != '') or (prediction != '' and ground_truth == ''):
            score = 0
        elif prediction == '' and ground_truth == '':
            score = 1
        elif len(prediction.split()) == 1 or len(ground_truth.split()) == 1:
            if prediction.split() == ground_truth.split():
                score = 1
            else:
                score = 0
        else:
            score = rouge_l.get_scores(prediction, ground_truth)[0]['rouge-l']['f']
        scores.append(score)

    return max(scores)

