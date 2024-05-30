import torch
import tqdm
import argparse
import time
import random
import requests
import numpy as np
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation

device = "cuda"

def main(args):
    
    dataset_port = {}
    output_dataset = {}
    tasks = []
    dataset_id = 0
    latency = []
    for dataset in args.datasets:
        dataset_port[dataset] = args.port[dataset_id]
        dataset_id += 1
        if dataset in ["nq_open", "GSM8K", "MedQUAD"]: #Common Knowledge QA
            prompts, labels, metrics = load_commonsense_evaluation(dataset, limit = args.limit)
        elif dataset in ["code2text", "dialog_summary", "cnn_news"]:
            prompts, labels, metrics = load_summarization_evaluation(dataset, limit = args.limit)
        elif dataset in ["triviaqa", "squad", "swde", "drop"]:
            prompts, labels, metrics = load_extraction_evaluation(dataset, limit = args.limit)
        else:
            print("No such dataset ", dataset)
        for i in range(len(prompts)):
            tasks.append([prompts[i], labels[i], 0, dataset])
        output_dataset[dataset]= {'output':[],'label':[], 'metrics':metrics}
    random.shuffle(tasks)
    port_batched = {}
    
    for port in args.port:
        port_batched[port] = []
    time_start = time.time()
    epoch = 0
    for task in tasks:
        time.sleep(max(0, time_start + epoch * args.interval_time - time.time()))
        epoch += 1
        
        port = dataset_port[task[-1]]
        task[2] = time.time()
        port_batched[port].append(task)
        if len(port_batched[port]) >= args.batch_size:
            sending_to_port = []
            for i in range(len(port_batched[port])):
                sending_to_port.append(port_batched[port][i][0])
            sending_to_port = {"prompts": sending_to_port}
            url = f"http://127.0.0.1:{port}/inference"
            response = requests.post(url, json = sending_to_port)
            response = response.json()
            for i in range(len(port_batched[port])):
                task = port_batched[port][i]
                output_dataset[task[-1]]['output'].append(response[str(i)])
                output_dataset[task[-1]]['label'].append(task[1])
                latency.append(time.time() - task[2])
            port_batched[port] = []
    for port in port_batched:
        if len(port_batched[port]) == 0:
            continue
        sending_to_port = []
        for i in range(len(port_batched[port])):
            sending_to_port.append(port_batched[port][i][0])
        sending_to_port = {"prompts": sending_to_port}
        url = f"http://127.0.0.1:{port}/inference"
        response = requests.post(url, json = sending_to_port)
        response = response.json()
        for i in range(len(port_batched[port])):
            task = port_batched[port][i]
            output_dataset[task[-1]]['output'].append(response[str(i)])
            output_dataset[task[-1]]['label'].append(task[1])
            latency.append(time.time() - task[2])
        port_batched[port] = []
        
    result = {}
    for dataset in output_dataset.keys():
        result[dataset] = [metric.compute(predictions = output_dataset[dataset]["output"], references = output_dataset[dataset]["label"]) for metric in output_dataset[dataset]["metrics"]]
    print(result)
    latency = np.array(latency)
    print("mean latency:", np.mean(latency))
    print("min:", np.min(latency), "25 per:", np.percentile(latency, 25),"median:", np.percentile(latency, 50), "75 per:", np.percentile(latency, 75),"max:", np.max(latency))
    

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Generate text using MambaLMHeadModel')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--datasets', type=str, nargs='+', default=['GSM8K'], help='Dataset name for commonsense loading')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples')
    parser.add_argument('--port', type = int, nargs='+', default = ['2001'], help='The model running on the end')
    parser.add_argument('--interval_time', type = float, nargs='+', default = 0.2, help='The model running on the end')
    args = parser.parse_args()

    main(args)