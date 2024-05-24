import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from datasets import load_metric, load_dataset
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation
import tqdm
import argparse


device = "cuda"
def main(args):
    batch_size = args.batch_size
    if args.model.find("mamba") != -1:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(ars.gmodel)
    model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    result = {}
    
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
            inputs = tokenizer.batch_encode_plus(prompts[i:end_index], padding = True, return_tensors='pt')
            inputs = inputs.to(device)
            input_batch = inputs.input_ids  # Get batch of input_ids
            outputs = model.generate(
                input_ids = input_batch,
                max_length=200,
            )
            # Decode and store generated text for each output in the batch
            for idx, output in enumerate(outputs):
                # Calculate the length of the input to skip it in the output
                input_length = input_batch[idx].size(0)
                generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)  # Skip the input part
                generated_texts.append(generated_text)
        result[dataset] = [metric.compute(predictions = generated_texts, references = labels) for metric in metrics]
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using MambaLMHeadModel')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--datasets', type=str, nargs='+', default=['GSM8K'], help='Dataset name for commonsense loading')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples')
    parser.add_argument('--model', type=str, default="state-spaces/mamba-2.8b", help='Model Name')

    args = parser.parse_args()

    main(args)