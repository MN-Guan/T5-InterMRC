import sys
sys.path.append('..')
import argparse
from transformers import T5Tokenizer, T5Config
from T5Model import T5ForMRC
import torch
from T5InterMRC import test
import json
from Auxiliary_Functions import (
    count_parameters, 
    set_seed, 
)
from eval_expmrc import evaluate_span


def get_pretrain(model_name):
    try:
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        config = T5Config.from_pretrained(model_name)
    except:
        print('Retry to get the tokenizer and config!')
        tokenizer, config = get_pretrain(model_name)
    return tokenizer, config

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='T5-InterMRC')
    parser.add_argument('--model_level', type=str, default='t5-large')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--state_file', type=str, default='./Data/squad_T5_829.pt')
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--dev_file_name', type=str, default='./Data/Datasets/ExpMRC/expmrc-squad-dev.json')
    parser.add_argument('--data_name', type=str, default='squad')
    parser.add_argument('--is_training', action='store_true')
    parser.add_argument('--prediction_file', type=str, default='./pred.json')
    
    args = parser.parse_args()

    device = torch.device(args.device)
    tokenizer, config = get_pretrain(args.model_level)
    model = T5ForMRC(config)
    print(f'The model has {count_parameters(model):,} parameters in total.')
    test(model, tokenizer, device, args)
    print(f'Done! The prediction results have been saved in the {args.prediction_file} file!')
