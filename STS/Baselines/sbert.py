"""
This examples trains BERT (or any other transformer model like RoBERTa, DistilBERT etc.) for the STSb and SICK datasets from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.
"""
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torch
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentenceTransformer,  LoggingHandler, losses, models, util
from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
from sentence_transformers.readers import InputExample
import logging
from datetime import datetime
import sys
sys.path.append('..')
sys.path.append('../..')
import os
import argparse
from Auxiliary_Functions import (
    time_since, 
    get_time_str, 
    count_parameters, 
    set_seed, 
    string_to_float,
)
from STS_preprocess import get_STS_examples

def convert_original_examples_to_input_examples(examples):
    new_examples = []
    for example in examples:
        inp_example = InputExample(texts=[example.sentence1, example.sentence2], label=example.gold_probability / 5.0)
        new_examples.append(inp_example)
    return new_examples

def load_data(args):
    original_train_examples = get_STS_examples(args.dataset_name, args.data_path, 'train', None, do_augment=False)
    original_eva_examples = get_STS_examples(args.dataset_name, args.data_path, 'dev', None, do_augment=False)
    original_test_examples = get_STS_examples(args.dataset_name, args.data_path, 'test', None, do_augment=False)
    train_examples = convert_original_examples_to_input_examples(original_train_examples)
    eva_examples = convert_original_examples_to_input_examples(original_eva_examples)
    test_examples = convert_original_examples_to_input_examples(original_test_examples)
    return train_examples, eva_examples, test_examples

def test(model, test_examples, args):
    ##############################################################################
    # Load the stored model and evaluate its performance on STS benchmark dataset
    ##############################################################################
    model = SentenceTransformer(args.result_file_name)
    test_evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_examples, name='sts-test')
    test_evaluator(model, output_path=args.result_file_name)

def train(args):
    train_examples, eva_examples, test_examples = load_data(args)
    # Use Huggingface/transformers model (like BERT, RoBERTa, XLNet, XLM-R) for mapping tokens to embeddings
    word_embedding_model = models.Transformer(args.model_name_or_path)
    # Apply mean pooling to get one fixed sized sentence vector
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                   pooling_mode_mean_tokens=True,
                                   pooling_mode_cls_token=False,
                                   pooling_mode_max_tokens=False)

    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=args.batch_size)
    train_loss = losses.CosineSimilarityLoss(model=model)

    logging.info("Read STS dev dataset")
    evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eva_examples, name='sts-dev')

    # Configure the training. We skip evaluation in this example
    warmup_steps = math.ceil(len(train_dataloader) * args.num_epochs  * args.warm_up_rate) #10% of train data for warm-up
    logging.info("Warmup-steps: {}".format(warmup_steps))

    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=evaluator,
              epochs=args.num_epochs,
              evaluation_steps=1000,
              warmup_steps=warmup_steps,
              output_path=args.result_file_name)
    test(model, test_examples, args)

if __name__ == '__main__':
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    #### /print debug information to stdout
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--model_name_or_path', type=str, default='bert-large-uncased')
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--batch_size',      type=int, default=20, help='Batch size after gradient accumulation.')
    parser.add_argument('--warm_up_rate', type=float, default=0.1, help='Ratio of warmup step in the whole training step.')
    parser.add_argument('--dataset_name', type=str, default='STSb')
    parser.add_argument('--data_path', type=str, default='./../Data/Datasets/', help='The path of datasets.')
    parser.add_argument('--result_file_name', type=str, default='./output/SBERT_States/', help='The path to save the training results.')
    args = parser.parse_args()
    set_seed(args.random_seed)
    train(args)


