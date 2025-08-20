from sentence_transformers import SentenceTransformer, util
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('..')
sys.path.append('../..')
from STS_preprocess import get_STS_examples
from scipy.stats import spearmanr, pearsonr
import tqdm
import argparse

def get_all_datasets(args):
    stsb_dev = get_STS_examples('STSb', args.data_path, 'dev', None, do_augment=False)
    stsb_test = get_STS_examples('STSb', args.data_path, 'test', None, do_augment=False)
    sick_dev = get_STS_examples('SICK', args.data_path, 'dev', None, do_augment=False)
    sick_test = get_STS_examples('SICK', args.data_path, 'test', None, do_augment=False)
    dataset_list = [stsb_dev, stsb_test, sick_dev, sick_test]
    return dataset_list

def get_one_dataset_result(examples, model_name_or_path, args):
    model = SentenceTransformer(model_name_or_path)
    model.to(args.device)
    pred_list = []
    gold_list = []
    for example in examples:
        s1 = example.sentence1
        s2 = example.sentence2
        gold_probability = example.gold_probability
        vectors = model.encode([s1, s2])
        pred_probability = util.cos_sim(vectors[0], vectors[1]).numpy().item()
        pred_list.append(pred_probability)
        gold_list.append(gold_probability / 5)
    return (round(spearmanr(pred_list, gold_list)[0] * 100, 2), round(pearsonr(pred_list, gold_list)[0] * 100, 2))

def get_one_model_result(dataset_list, model_name, args):
    complete_model_path = os.path.join(args.model_path, model_name)
    dataset_name_list = ['stsb_dev', 'stsb_test', 'sick_dev', 'sick_test']
    data_tqdm = tqdm.tqdm(range(len(dataset_list)), desc = 'Processing Data')
    with open(args.baseline_results_file_name, 'a', encoding = 'utf-8') as f:
        for i in data_tqdm:
            dataset = dataset_list[i]
            dataset_name = dataset_name_list[i]
            spearman_score, pearson_score = get_one_dataset_result(dataset, complete_model_path, args)
            f.write(f'----{dataset_name}----{model_name}----\n')
            f.write('pearson_coefficient:{:.2f}, spearman_coefficient:{:.2f}\n'.format(pearson_score, spearman_score))
        f.write('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='./../Data/Model_States/')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--data_path', type=str, default='./../Data/Datasets/', help='The path of datasets.')
    parser.add_argument('--baseline_results_file_name', type=str, default='./bert_cls_results.txt', help='The path to save result.')
    # model_name_list = ['bert-cls-base', 'bert-cls-large', 'bert-mean-base', 'bert-mean-large', 'bert-nli-base', 'bert-nli-large', 'sbert-nli-base', 'sbert-nli-large', 'sroberta-nli-base', 'sroberta-nli-large']
    model_name_list = ['bert-cls-base', 'bert-cls-large']
    model_tqdm = tqdm.tqdm(model_name_list, desc = 'Processing Model')
    args = parser.parse_args()
    dataset_list = get_all_datasets(args)
    for model_name in model_tqdm:
        get_one_model_result(dataset_list, model_name, args)
    
