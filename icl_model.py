from semantic_router import Route
import os
import argparse
from getpass import getpass
from semantic_router.layer import RouteLayer
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation

def ICL(prompt, dataset = "nq_open", limit = 2, tokenizer = "EleutherAI/gpt-neox-20b"):
    if not prompt:
        return None
    if dataset in ["nq_open", "GSM8K", "MedQUAD"]: #Common Knowledge QA
        prompts, labels, _ = load_commonsense_evaluation(dataset, limit = limit)
    elif dataset in ["code2text", "dialog_summary", "cnn_news"]:
        prompts, labels, __annotations__ = load_summarization_evaluation(dataset, limit = limit)
    elif dataset in ["triviaqa", "squad", "swde", "drop"]:
        prompts, labels, _ = load_extraction_evaluation(dataset, limit = limit)
    else:
        print("No such dataset ", dataset)
    examples = [(prompts[i].replace("Answer:", ""), labels[i]) for i in range(limit)]
    print(examples)
    prefixed_prompt = create_prompt(examples, final_question=args.prompt)
    print(prefixed_prompt)
    return prefixed_prompt

def create_prompt(examples, final_question):
    # Initialize the prompt with an introduction
    prompt = "In this prompt, I'm providing examples of question and answer pairs. Please consider these examples before responding to the question that follows.\n\n"

    # Add each example to the prompt
    for i, (question, answer) in enumerate(examples, 1):
        prompt += f"Example {i}:\nQ: {question}\nA: {answer}\n\n"

    # Add the final question to the prompt
    prompt += f"Based on the above examples, now I would like to ask:\nQ: {final_question}\n"

    return prompt


def main(args):
    n_shot_prompt = ICL(args.prompt, args.dataset, args.limit, args.tokenizer)
    if not n_shot_prompt:
        print(" ================ Missing Prompt ================")
        return None
    return n_shot_prompt



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default = "nq_open")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--limit", type=int, default = 2)
    parser.add_argument('--prompt', type=str, help='A prompt string')
    args = parser.parse_args()

    main(args)


