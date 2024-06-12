import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM
from datasets import load_metric, load_dataset
from loaddata.commonsense import load_commonsense_evaluation, load_commonsense_training_prompts
from loaddata.summarization import load_summarization_evaluation, load_summarization_training_prompts
from loaddata.informationextraction import load_extraction_evaluation, load_extraction_training_prompts
import tqdm
import argparse
import time
import os
from routing import get_embeddings_dataset_list, calculate_max_similarities

device = "cuda"
def main(args):
    if device == "cuda":
        torch.cuda.empty_cache()
        
    batch_size = args.batch_size

    if args.model.find("state-spaces/mamba") != -1:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ##load models
    commonsense_model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    commonsense_model.load_state_dict(torch.load("saved_models/state-spaces_mamba-790m_commonsense/checkpoint-1600/pytorch_model.bin"))
    summarization_model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    summarization_model.load_state_dict(torch.load("saved_models/state-spaces_mamba-790m_summarization/checkpoint-1600/pytorch_model.bin"))
    extraction_model = MambaLMHeadModel.from_pretrained(args.model, device=device, dtype=torch.float16)
    extraction_model.load_state_dict(torch.load("saved_models/state-spaces_mamba-790m_extraction/checkpoint-1800/pytorch_model.bin"))
    print("loaded all submodels from checkpoint")

    ## setting up router
    commonsense_prompts = load_commonsense_training_prompts(args.vectorNum)
    summarization_prompts = load_summarization_training_prompts(tokenizer, args.vectorNum)
    extraction_prompts = load_extraction_training_prompts(tokenizer, args.vectorNum)
    training_prompts = commonsense_prompts + summarization_prompts + extraction_prompts
    router_embedding = get_embeddings_dataset_list(training_prompts)
    print("got training embeddings")

    result = {}
    
    for dataset in args.datasets:
        if dataset in ["nq_open", "GSM8K", "MedQUAD"]: #Common Knowledge QA
            prompts, labels, metrics = load_commonsense_evaluation(dataset, limit = args.limit, tokenizer = tokenizer)
        elif dataset in ["code2text", "dialog_summary", "cnn_news"]:
            prompts, labels, metrics = load_summarization_evaluation(dataset, limit = args.limit, tokenizer = tokenizer)
        elif dataset in ["triviaqa", "squad", "swde", "drop"]:
            prompts, labels, metrics = load_extraction_evaluation(dataset, limit = args.limit, tokenizer = tokenizer)
        else:
            print("No such dataset ", dataset)

        ## route
        routed_prompts = {}
        routed_prompts["commonsense"] = []
        routed_prompts["summarization"] = []
        routed_prompts["extraction"] = []
        routed_labels = {}
        routed_labels["commonsense"] = []
        routed_labels["summarization"] = []
        routed_labels["extraction"] = []
        for i, (p, l) in enumerate(zip(prompts, labels)):
            model_index = calculate_max_similarities(p, router_embedding, n = args.vectorNum)
            if model_index <= 3:
                routed_prompts["commonsense"].append(p)
                routed_labels["commonsense"].append(l)
                print(i, "routed to commonsense")
            elif model_index <= 6:
                routed_prompts["summarization"].append(p)
                routed_labels["summarization"].append(l)
                print(i, "routed to summarization")
            else:
                routed_prompts["extraction"].append(p)
                routed_labels["extraction"].append(l)
                print(i, "routed to extraction")

        #option: nq_open,  GSM8K, MedQUAD
        generated_texts = []
        reordered_labels = routed_labels["commonsense"] + routed_labels["summarization"] + routed_labels["extraction"]

        for submodel in ["commonsense", "summarization", "extraction"]:
            cur_prompts = routed_prompts[submodel]
            print("computing dataset", dataset, "using submodel", submodel, "for", len(cur_prompts), "prompts")

            # Iterate over tokenized inputs
            for i in tqdm.tqdm(range(0, len(cur_prompts), batch_size), desc="Generating outputs"):
                # Determine the size of the current batch (it may be less than batch_size at the end)
                end_index = min(i + batch_size, len(cur_prompts))
                inputs = tokenizer.batch_encode_plus(cur_prompts[i:end_index], padding = True, return_tensors='pt', truncation = True, max_length = 200)
                inputs = inputs.to(device)
                input_batch = inputs.input_ids  # Get batch of input_ids
                max_len = 300

                if submodel == "commonsense":
                    outputs = commonsense_model.generate(
                        input_ids = input_batch,
                        max_length=max_len
                    )
                elif submodel == "summarization":
                    outputs = summarization_model.generate(
                        input_ids = input_batch,
                        max_length=max_len
                    )
                else:
                    outputs = extraction_model.generate(
                        input_ids = input_batch,
                        max_length=max_len
                    )

                # Decode and store generated text for each output in the batch
                for idx, output in enumerate(outputs):
                    # Calculate the length of the input to skip it in the output
                    input_length = input_batch[idx].size(0)
                    generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)  # Skip the input part
                    generated_texts.append(generated_text)
        
        result[dataset] = [metric.compute(predictions = generated_texts, references = reordered_labels) for metric in metrics]
    print(result)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using MambaLMHeadModel')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for generation')
    parser.add_argument('--datasets', type=str, nargs='+', default=['GSM8K'], help='Dataset name for commonsense loading')
    parser.add_argument('--limit', type=int, default=500, help='Limit the number of samples')
    parser.add_argument('--model', type=str, default="state-spaces/mamba-790m", help='Model Name')
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--vectorNum', type=int, default=3, help='The number of vector embeddings per dataset')
    args = parser.parse_args()

    main(args)