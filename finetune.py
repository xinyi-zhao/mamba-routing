import torch
import argparse

from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer, TrainingArguments
from loaddata.commonsense import load_commonsense_training
from loaddata.informationextraction import load_extraction_training
from loaddata.summarization import load_summarization_training

from training.mamba_trainer import MambaTrainer
from transformers import Trainer

def get_name(name):
    name = name.replace("/",'_')
    return name

def run(args):
    if args.model.find("mamba") != -1:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
    else:
        tokenizer = AutoTokenizer.from_pretrained(ars.gmodel)
    if args.dataset in ["nq_open", "GSM8K", "MedQUAD", "commonsense"]: #Common Knowledge QA
        data_module = load_commonsense_training(args.dataset, tokenizer, limit = args.limit)
    elif args.dataset in ["code2text", "dialog_summary", "cnn_news", "summarization"]:
        data_module = load_summarization_training(args.dataset, tokenizer, limit = args.limit)
    elif args.dataset in ["triviaqa", "squad", "swde", "drop", "extraction"]:
        data_module = load_extraction_training(args.dataset, tokenizer, limit = args.limit)
    else:
        print("No such dataset ", dataset)
    
    # data_module = ChatDataModule(
    #     tokenizer=tokenizer,
    #     data_path=args.data_path,
    #     conversation_template=tokenizer.chat_template,
    #     max_tokens=2048
    # )

    model = MambaLMHeadModel.from_pretrained(args.model, dtype=torch.bfloat16, device="cuda")
    trainer = MambaTrainer(
        model=model,
        train_dataset=data_module.dataset,
        tokenizer=tokenizer,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_epochs,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            optim=args.optim,
            output_dir=f"{args.save_path}/{get_name(args.model)}_{get_name(args.dataset)}",
            logging_steps=50,
            save_steps=500,
        ),
        data_collator=data_module.data_collator,
    )

    trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="state-spaces/mamba-370m")
    parser.add_argument("--dataset", type=str, default = "nq_open")
    parser.add_argument("--tokenizer", type=str, default="EleutherAI/gpt-neox-20b")
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--data_path", type=str, default="./data/ultrachat_small.jsonl")
    parser.add_argument("--num_epochs", type=int, default=1)
    parser.add_argument("--save_path", type=str, default = "saved_models")
    parser.add_argument("--limit", type=int, default = -1)
    args = parser.parse_args()

    run(args)
