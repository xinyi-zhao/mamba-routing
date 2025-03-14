import torch
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation
import tqdm
import argparse

from composed_model import ComposedModel

device = "cuda"
def main(args):
    batch_size = args.batch_size
    result = {}

    ## Define composed model
    composed = ComposedModel(device)
    composed.add_model("code2text", "mrm8488/mamba-coder", in_context_prompt = "", checkpoint = None)
    composed.finalize_model()
    
    for dataset in args.datasets:
        if dataset in ["nq_open", "GSM8K", "MedQUAD"]: #Common Knowledge QA
            prompts, labels, metrics = load_commonsense_evaluation(dataset, limit = args.limit)
        elif dataset in ["code2text", "dialog_summary", "cnn_news"]:
            prompts, labels, metrics = load_summarization_evaluation(dataset, limit = args.limit)
        elif dataset in ["triviaqa", "squad", "swde", "drop"]:
            prompts, labels, metrics = load_extraction_evaluation(dataset, limit = args.limit)
        else:
            print("No such dataset ", dataset)
        #option: nq_open,  GSM8K, MedQUAD
        generated_texts = []
        # Iterate over tokenized inputs
        for i in tqdm.tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):

            # Determine the size of the current batch (it may be less than batch_size at the end)
            end_index = min(i + batch_size, len(prompts))
            generated = composed.batch(prompts[i:end_index])
            generated_texts.extend(generated)

        result[dataset] = [metric.compute(predictions = generated_texts, references = labels) for metric in metrics]
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using MambaLMHeadModel')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size for generation')
    parser.add_argument('--datasets', type=str, nargs='+', default=['GSM8K'], help='Dataset name for commonsense loading')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples')

    args = parser.parse_args()

    main(args)