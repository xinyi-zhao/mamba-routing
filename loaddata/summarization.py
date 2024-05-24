import pandas as pd
from datasets import load_metric, load_dataset
import numpy as np
from evaluate import load
import re
def load_summarization_evaluation(name = "nq", limit = -1):
    if(name == "code2text"):
        dataset = load_dataset("google/code_x_glue_ct_code_to_text", "java")
        df = dataset["validation"].to_pandas() 
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = row["original_string"] + "What is the funtion of this code? Answer:"
            label = row["docstring"]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [load("rouge")]
        return prompts, labels, metrics
    elif name == "dialog_summary":
        dataset = load_dataset("knkarthick/dialogsum")
        df = dataset["validation"].to_pandas() 
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = "Dialogue:" + row["dialogue"] + "What is the summarization of this dialogue? Answer:"
            label = row["summary"]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [load("rouge")]
        return prompts, labels, metrics
    elif name == "cnn_news":
        dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")
        df = dataset["validation"].to_pandas() 
        prompts = []
        labels = []
        for _ , row in df.iterrows():
            prompt = "Article: " + row["article"] + "\nWhat is the highlight of this article? Answer:"
            label = row["highlights"]
            prompts.append(prompt)
            labels.append(label)
            if limit > -1 and len(prompts) == limit:
                break
        metrics = [load("rouge")]
        return prompts, labels, metrics