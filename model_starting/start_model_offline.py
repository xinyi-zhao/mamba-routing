import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import GPTNeoForCausalLM, GPTNeoXForCausalLM, OPTForCausalLM
from datasets import load_metric, load_dataset
import tqdm
import argparse
import time
from fastapi import FastAPI, Query, Body, Request
from fastapi import File, UploadFile
from fastapi.logger import logger as fastapi_logger
import uvicorn
from pydantic import BaseModel, Field
from logging.handlers import RotatingFileHandler
import logging
import os
from datetime import datetime
app = FastAPI()
device = "cuda"
model = None
tokenizer = None

class InferenceData(BaseModel):
    prompts: list[str] = Field(..., example=["Hello, world!", "How are you?"])


@app.post('/inference')
async def inference(data: InferenceData):
    # Parse JSON data from the request
    # save the image
    global tokenizer, model
    inputs = tokenizer.batch_encode_plus(data.prompts, padding = True, return_tensors='pt')
    inputs = inputs.to(device)
    input_batch = inputs.input_ids  # Get batch of input_ids
    
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
        # Decode and store generated text for each output in the batch
    result = {}
    for idx, output in enumerate(outputs):
        # Calculate the length of the input to skip it in the output
        input_length = input_batch[idx].size(0)
        generated_text = tokenizer.decode(output[input_length:], skip_special_tokens=True)  # Skip the input part
        result[idx] = generated_text
    
    return result

def main(args):
    DEBUG = True
    global tokenizer, model
    formatter = logging.Formatter(
    "[%(asctime)s.%(msecs)03d] %(levelname)s [%(thread)d] - %(message)s", "%Y-%m-%d %H:%M:%S")
    # create log file
    if not os.path.exists('./dblog'):
        os.makedirs('./dblog')
    # touch new log file based on current datetime
    log_file = datetime.now().strftime('./dblog/%Y-%m-%d_%H-%M-%S.log')
    handler = RotatingFileHandler(log_file, backupCount=0)
    logging.getLogger().setLevel(logging.NOTSET)
    fastapi_logger.addHandler(handler)
    handler.setFormatter(formatter)

    if args.model.find("state-spaces/mamba") != -1:
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
        
    fastapi_logger.info('****************** Starting Server *****************')
    host = '127.0.0.1' if DEBUG else '0.0.0.0'
    uvicorn.run(app, host=host, port=args.port)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate text using MambaLMHeadModel')
    parser.add_argument('--model', type=str, default="state-spaces/mamba-370m", help='Model Name')
    parser.add_argument('--checkpoint', type=str, default="")
    parser.add_argument('--port', type = int, default =2001)
    args = parser.parse_args()

    main(args)