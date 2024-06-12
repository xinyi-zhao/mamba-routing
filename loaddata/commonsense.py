import pandas as pd
from datasets import load_metric, load_dataset
import numpy as np
from evaluate import load
import re
import gc
from .finetunedataset import FinetuneDataModumn

import random

class contains_metric:
    def compute(predictions, references):
        contains = []
        for pred, label in zip(predictions, references):
            pred = pred.replace(",","")
            contains.append(int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), pred))))
        return 1.0 * np.array(contains).sum()/len(contains)
class squad_metric:
    def compute(predictions, references ):
        metric = load_metric("squad", trust_remote_code=True)
        preds = []
        refs = []
        id = 0
        for pred,label in zip(predictions,references):
            id += 1
            preds.append({'prediction_text': pred, 'id': str(id)})
            refs.append({'answers': {'answer_start': [0], 'text': [label]}, 'id': str(id)})
        results = metric.compute(predictions=preds, references=refs)
        return results
def load_commonsense_evaluation(name = "nq", limit = -1, tokenizer = None):
    if(name == "nq_open"):
        dataset = load_dataset("nq_open")
        df = dataset["validation"].to_pandas() 
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = row["question"] + " Answer:"
            label = row["answer"][0]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [squad_metric]
        return prompts, labels, metrics
    
    elif(name == "GSM8K"):
        dataset = load_dataset("gsm8k", "main")
        df = dataset["test"].to_pandas()
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = row["question"] 
            label = row["answer"].split("#### ")[-1]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [contains_metric]
        return prompts, labels, metrics
    
    elif (name == "MedQUAD"):
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        df = dataset["train"].to_pandas()[-1000:]
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = row["Question"]
            label = row["Answer"]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [load("rouge")]
        return prompts, labels, metrics
    return None, None

def load_commonsense_training(name, tokenizer, limit = -1):
    prompts = []
    labels = []
    if(name == "nq_open" or name == "commonsense"):
        dataset = load_dataset("nq_open")
        df = dataset["train"].to_pandas() 
        cnt = 0
        for _ , row in df.iterrows():
            prompt = row["question"] + " Answer:"
            label = row["answer"][0]
            prompts.append(prompt)
            labels.append(label)
            cnt += 1
            if limit > -1 and cnt == limit:
                break
        print("loaded", cnt, "of nq_open")
    
    if(name == "GSM8K" or name == "commonsense"):
        dataset = load_dataset("gsm8k", "main")
        df = dataset["train"].to_pandas()
        cnt = 0
        for _ , row in df.iterrows():
            prompt = row["question"] 
            label = row["answer"]
            prompts.append(prompt)
            labels.append(label)
            cnt += 1
            if limit > -1 and cnt == limit:
                break
        print("loaded", cnt, "of GSM8K")
    
    if(name == "MedQUAD" or name == "commonsense"):
        dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
        df = dataset["train"].to_pandas()[:-1000]
        cnt = 0
        for _ , row in df.iterrows():
            cnt += 1
            prompt = row["Question"]
            label = row["Answer"]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and cnt == limit:
                break
        print("loaded", cnt, "of MedQUAD")

    binded = list(zip(prompts, labels))
    random.shuffle(binded)
    prompts, labels = zip(*binded)

    return FinetuneDataModumn(tokenizer, prompts, labels)


def load_commonsense_training_prompts(limit = -1):
    prompts = []
    
    dataset = load_dataset("nq_open")
    df = dataset["train"].to_pandas() 
    cnt = 0
    for _ , row in df.iterrows():
        prompt = row["question"] + " Answer:"
        prompts.append(prompt)
        cnt += 1
        if limit > -1 and cnt == limit:
            break
    
    dataset = load_dataset("gsm8k", "main")
    df = dataset["train"].to_pandas()
    cnt = 0
    for _ , row in df.iterrows():
        prompt = row["question"]
        prompts.append(prompt)
        cnt += 1
        if limit > -1 and cnt == limit:
            break

    dataset = load_dataset("keivalya/MedQuad-MedicalQnADataset")
    df = dataset["train"].to_pandas()[:-1000]
    cnt = 0
    for _ , row in df.iterrows():
        cnt += 1
        prompt = row["Question"]
        prompts.append(prompt)
        if limit > -1 and cnt == limit:
            break

    return prompts