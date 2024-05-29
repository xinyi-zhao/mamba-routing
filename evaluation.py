import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM
from datasets import load_metric, load_dataset
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation
import tqdm
import argparse
import time


device = "cuda"
def main(args):
    batch_size = args.batch_size

    if args.model.find("mamba") != -1:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.model.find("mamba") != -1:
        model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    elif args.model.find("gpt-neo") != -1:
        model = GPTNeoForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    elif args.model.find("pythia") != -1:
        model = GPTNeoXForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    elif args.model.find("opt") != -1:
        model = OPTForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16).cuda()
    else:
        print("!!! model not supported")
    print(">>>>>>>>>>>>>> loaded pretrained from", args.model)

    if (args.checkpoint != ""):
        model.load_state_dict(torch.load(f'{args.checkpoint}/pytorch_model.bin'))
        print(">>>>>>>>>>>>>> loaded checkpoint from", args.checkpoint)
    
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
        time_taken = 0
        input_tokens = 0
        output_tokens = 0
        # Iterate over tokenized inputs
        for i in tqdm.tqdm(range(0, len(prompts), batch_size), desc="Generating outputs"):
            # Determine the size of the current batch (it may be less than batch_size at the end)
            end_index = min(i + batch_size, len(prompts))
            inputs = tokenizer.batch_encode_plus(prompts[i:end_index], padding = True, return_tensors='pt')
            inputs = inputs.to(device)
            input_batch = inputs.input_ids  # Get batch of input_ids
            start = time.time()
            if args.model.find("gpt-neo") != -1 or args.model.find("pythia") != -1:
                outputs = model.generate(
                    input_ids = input_batch,
                    max_length=200,
                    pad_token_id=tokenizer.pad_token_id,
                )
            else:
                outputs = model.generate(
                    input_ids = input_batch,
                    max_length=200
                )
            end = time.time()
            time_taken += end - start 
            # Decode and store generated text for each output in the batch
            for idx, output in enumerate(outputs):
                # Calculate the length of the input to skip it in the output
                input_length = input_batch[idx].size(0)
                generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)  # Skip the input part
                generated_texts.append(generated_text)
                input_tokens += input_length
                output_tokens += len(output) - input_length
        result[dataset] = [metric.compute(predictions = generated_texts, references = labels) for metric in metrics]
        print("dataset: %s\nlatency: %f\nnumber of input tokens: %d\nnumber of output tokens: %d\ntime per ouput token: %fns" % (dataset, time_taken, input_tokens, output_tokens, (time_taken/output_tokens)*(10**9)))
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using MambaLMHeadModel')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for generation')
    parser.add_argument('--datasets', type=str, nargs='+', default=['GSM8K'], help='Dataset name for commonsense loading')
    parser.add_argument('--limit', type=int, default=-1, help='Limit the number of samples')
    parser.add_argument('--model', type=str, default="state-spaces/mamba-370m", help='Model Name')
    parser.add_argument('--checkpoint', type=str, default="")
    args = parser.parse_args()

    main(args)