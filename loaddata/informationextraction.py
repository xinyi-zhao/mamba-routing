import pandas as pd
from datasets import load_metric, load_dataset
import numpy as np
from evaluate import load
import re

class contains_metric:
    def compute(predictions, references):
        contains = []
        for pred, label in zip(predictions, references):
            contains.append(int(bool(re.search(re.compile(re.escape(label), re.IGNORECASE), pred))))
        return 1.0 * np.array(contains).sum()/len(contains)
def load_extraction_evaluation(name = "nq",limit = -1):
    if(name == "triviaqa"):
        dataset = load_dataset("TimoImhof/TriviaQA-in-SQuAD-format")
        df = dataset["unmodified"].to_pandas()[-1000:]
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = row["context"] + row["question"] + "Answer:"
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
            prompt = row["context"] + row["question"] + "Answer:"
            label = row["answers"]["text"][0]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [contains_metric]
        return prompts, labels, metrics
    elif name == "swde":
        dataset = load_dataset("hazyresearch/based-swde")
        df = dataset["validation"].to_pandas()
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = row["text"]
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
            prompt = row["passage"] + row["question"] + "Answer:"
            label = row["answers_spans"]["spans"][0]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [contains_metric]
        return prompts, labels, metrics