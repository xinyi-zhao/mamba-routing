import torch
import tqdm
import argparse
import time
import random
import requests
import datetime
import numpy as np
import os
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation
import multiprocessing


class TaskProcessor(multiprocessing.Process):
    def __init__(self, task_queue, batch_size, port, logtime):
        super().__init__()
        self.task_queue = task_queue
        self.batch_size = batch_size  # Number of tasks to accumulate before processing
        self.port = port
        self.results = {}
        self.latency = []
        self.last_id = -1
        self.logtime = logtime
    
    def run(self):
        while True:
            cnt = 0
            length = len(self.task_queue)
            ending_pos = -1
            for i in range(self.last_id + 1, length):
                if self.task_queue[i]["port"] == self.port:
                    cnt += 1
                    if(cnt == self.batch_size):
                        ending_pos = i + 1
                        break
            if ending_pos == -1:
                ending_pos = length
                
            if cnt >= self.batch_size or (ending_pos == length and self.task_queue[length - 1]["port"] == -1):
                self.call_server(self.last_id + 1, ending_pos)
                self.last_id = ending_pos - 1
                
            if self.task_queue[length - 1]["port"] == -1 and ending_pos == length:
                self.get_result()
                break
    
    def call_server(self, st, ed):
        print("port:",self.port, "starting_id:",st,"ending_id:",ed)
        sending_to_port = []
        for i in range(st, ed):
            if self.task_queue[i]["port"] == self.port:
                sending_to_port.append(self.task_queue[i]["prompt"])
        if len(sending_to_port) == 0:
            return 
        sending_to_port = {"prompts": sending_to_port}
        url = f"http://127.0.0.1:{self.port}/inference"
        response = requests.post(url, json = sending_to_port)
        response = response.json()
        response_id = 0
        for i in range(st, ed):
            if self.task_queue[i]["port"] == self.port:
                task = self.task_queue[i]
                dataset = task["dataset"]
                if dataset not in self.results.keys():
                    self.results[dataset] = {'output':[],'label':[], 'metrics':task["metrics"]}
                self.results[dataset]["output"].append(response[str(response_id)])
                self.results[dataset]["label"].append(task["label"])
                self.latency.append(time.time() - task["start_time"])
                response_id += 1
    
    def get_result(self):
        for dataset in self.results.keys():
            ans = [metric.compute(predictions = self.results[dataset]["output"], references = self.results[dataset]["label"]) for metric in self.results[dataset]["metrics"]]
            print(dataset, ans)
        latency = np.array(self.latency)
        print("port:",self.port, "mean latency:", np.mean(latency), "min:", np.min(latency), "25 per:", np.percentile(latency, 25),"median:", np.percentile(latency, 50), "75 per:", np.percentile(latency, 75),"max:", np.max(latency))
        np.save(f"latency_results/{self.logtime}/latency_{self.port}.npy", latency)
    

def task_generator(tasks, task_queue, interval_time):
    for task in tasks:
        task["start_time"] = time.time()
        task_queue.append(task)
        time.sleep(interval_time)
    task_queue.append({"port": -1})

def main(args):
    tasks = []
    dataset_id = 0
    for dataset in args.datasets:
        port = args.port[dataset_id]
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
            tasks.append({"prompt":prompts[i], "label":labels[i], "dataset":dataset, "metrics": metrics, "port":port})
    
    random.shuffle(tasks)
    logtime = datetime.datetime.now().strftime("%Y%m%d_%H%M%S").replace(" ", "_").replace("/", "_").replace("\\", "_")
    os.mkdir(f"latency_results/{logtime}")
    print("save_dir:", logtime)
    
    manager = multiprocessing.Manager()
    task_queue = manager.list()
    
    generator_process = multiprocessing.Process(target=task_generator, args=(tasks, task_queue, args.interval_time))
    generator_process.start()

    # Start processors
    unique_ports = np.unique(np.array(args.port))
    processors = []
    for port in unique_ports:
        processors.append(TaskProcessor(task_queue, args.batch_size, port, logtime))
    
    for processor in processors:
        processor.start()

    # Wait for the generator process to finish (it runs indefinitely until interrupted)
    generator_process.join()
    for processor in processors:
        processor.join()
     

if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description='Generate text using MambaLMHeadModel')
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--datasets', type=str, nargs='+', default=['GSM8K'], help='Dataset name for commonsense loading')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples')
    parser.add_argument('--port', type = int, nargs='+', default = ['2001'], help='The model running on the end')
    parser.add_argument('--interval_time', type = float, default = 0.2, help='The model running on the end')
    args = parser.parse_args()

    main(args)