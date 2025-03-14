import pandas as pd
from datasets import load_metric, load_dataset
import numpy as np
from evaluate import load
import re
from .finetunedataset import FinetuneDataModumn

import random

truncate_len = 400

def truncate(context, tokenizer, length = truncate_len):
    tokens = tokenizer.encode(context, return_tensors = "pt")[0]
    tokens = tokens[:min(length, len(tokens))]
    context = tokenizer.decode(tokens)
    return context

class contains_metric:
    def compute(predictions, references):
        contains = []
        for pred, label in zip(predictions, references):
            contains.append(int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), pred))))
        return 1.0 * np.array(contains).sum()/len(contains)
def load_extraction_evaluation(name = "nq",limit = -1, tokenizer = None):
    if(name == "triviaqa"):
        dataset = load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")
        df = dataset["unmodified"].to_pandas()[-1000:]
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = truncate(row["context"], tokenizer, truncate_len) + row["question"] + "Answer:"
            label = row["answers"]["text"][0]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [contains_metric]
        return prompts, labels, metrics
    elif name == "squad":
        dataset = load_dataset("rajpurkar/squad")
        df = dataset["validation"].to_pandas()
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = truncate(row["context"], tokenizer, truncate_len) + row["question"] + "Answer:"
            label = row["answers"]["text"][0]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [contains_metric]
        return prompts, labels, metrics
    elif name == "swde":
        dataset = load_dataset("hazyresearch/based-swde")
        df = dataset["validation"].to_pandas()[-300:]
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt =  truncate(row["text"], tokenizer, truncate_len)
            label = row["value"]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [contains_metric]
        return prompts, labels, metrics
    elif name == "drop":
        dataset = load_dataset("ucinlp/drop")
        df = dataset["validation"].to_pandas()
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            if (row["answers_spans"]["types"][0] != 'span'):
                continue
            prompt =  truncate(row["passage"], tokenizer, truncate_len) + row["question"] + "Answer:"
            label = row["answers_spans"]["spans"][0]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [contains_metric]
        return prompts, labels, metrics

def load_extraction_training(name ,tokenizer, limit = -1):
    prompts = []
    labels = []
    if(name == "triviaqa" or name =="extraction"):
        dataset = load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")
        df = dataset["unmodified"].to_pandas()[:-1000]
        cnt = 0
        for _ , row in df.iterrows():
            prompt = truncate(row["context"], tokenizer, truncate_len) + row["question"] + "Answer:"
            label = row["answers"]["text"][0]
            prompts.append(prompt)
            labels.append(label)
            cnt += 1
            if limit > -1 and cnt == limit:
                break

    if name == "squad" or name =="extraction":
        dataset = load_dataset("rajpurkar/squad")
        df = dataset["train"].to_pandas()
        cnt = 0
        for _ , row in df.iterrows():
            prompt = truncate(row["context"], tokenizer, truncate_len) + row["question"] + "Answer:"
            label = row["answers"]["text"][0]
            cnt += 1
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and cnt == limit:
                break

    if name == "swde" or name =="extraction":
        dataset = load_dataset("hazyresearch/based-swde")
        df = dataset["validation"].to_pandas()[:-300]
        cnt = 0
        for _ , row in df.iterrows():
            prompt = truncate(row["text"], tokenizer, truncate_len)
            label = row["value"]
            prompts.append(prompt)
            labels.append(label)
            cnt += 1
            if limit > -1 and cnt == limit:
                break
 
    if name == "drop" or name =="extraction":
        dataset = load_dataset("ucinlp/drop")
        df = dataset["train"].to_pandas()
        cnt = 0
        for _ , row in df.iterrows():
            if (row["answers_spans"]["types"][0] != 'span'):
                continue
            prompt = truncate(row["passage"], tokenizer, truncate_len) + row["question"] + "Answer:"
            label = row["answers_spans"]["spans"][0]
            prompts.append(prompt)
            labels.append(label)
            cnt += 1
            if limit > -1 and cnt == limit:
                break

    binded = list(zip(prompts, labels))
    random.shuffle(binded)
    prompts, labels = zip(*binded)
    
    return FinetuneDataModumn(tokenizer, prompts, labels)