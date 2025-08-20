import logging
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
import collections
import re
import string
import tensorflow_datasets as tfds
from typing import Optional, Tuple
warnings.filterwarnings('ignore')

import transformers
from transformers import T5ForConditionalGeneration
from transformers import T5Tokenizer
from STS.STS_preprocess import STS_example, STS_convert_examples_to_features
from Auxiliary_Functions import (
    get_time_str, 
    count_parameters, 
    string_to_float,
)
from Preprocess import (
    mycollate_eva,
)


def load_data(sentence1, sentence2, tokenizer, verbose):
    gold_probability = None
    sts_id = None
    example = [STS_example(sentence1, sentence2, gold_probability, sts_id)]
    eva_feature = STS_convert_examples_to_features(examples=example, 
                                                   tokenizer=tokenizer,
                                                   max_seq_length=None,
                                                   is_training=False,
                                                   do_divide=False,
                                                   verbose=verbose)
    return eva_feature, example


def inference(model, tokenizer, device, eva_iterator, eva_examples, eva_features, verbose):
    model.eval()
    num_beams = 1
    max_output_length = 10
    if verbose:
        epoch_iterator = tqdm.tqdm(eva_iterator, desc='Evaluate')
    else:
        epoch_iterator = eva_iterator
    batch_size = eva_iterator.batch_size
    predictions = dict()
    for iterator_index, batch in enumerate(epoch_iterator):
        batch = tuple(item.to(device) for item in batch.values())
        outputs = model.generate(input_ids=batch[0],
                                 attention_mask=batch[1],
                                 num_beams=num_beams,
                                 max_length=max_output_length,
                                 # early_stopping=True
        )
        for i in range(batch[0].shape[0]):
            pred = tokenizer.decode(outputs[i], skip_special_tokens=True)
            pred = pred.replace(' ', '')
            pred = string_to_float(pred)
            sts_id = eva_features[i + iterator_index * batch_size]['sts_id']
            predictions[sts_id] = predictions.get(sts_id, [])
            predictions[sts_id].append(pred)
    return predictions


def test(model, eva_features, eva_examples, batch_size, device, tokenizer, verbose):
    eva_iterator = torch.utils.data.DataLoader(eva_features, 
                                               collate_fn=mycollate_eva, 
                                               batch_size=batch_size,
                                               shuffle=False)
    pred = inference(model, tokenizer, device, eva_iterator, eva_examples, eva_features, verbose)
    return pred
