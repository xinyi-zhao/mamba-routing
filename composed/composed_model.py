from langchain_core.prompts import PromptTemplate

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from langchain_core.runnables import RunnableLambda, RunnablePassthrough


## prompt that simply forwards

template = """{question}"""
prompt = PromptTemplate.from_template(template)

### mamba-coder

device = "cuda"
model_name = "mrm8488/mamba-coder"
eos_token = "<|endoftext|>"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.eos_token = eos_token
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

mamba_coder_model = MambaLMHeadModel.from_pretrained(
        model_name, device=device, dtype=torch.float16)


def mamba_coder(prompt):
  #print("mamba-coder start==========")
  messages = []
  prompt = prompt.to_string()
  messages.append(dict(role="user", content=prompt))

  input_ids = tokenizer.apply_chat_template(
              messages, return_tensors="pt", add_generation_prompt=True
  ).to(device)

  out = mamba_coder_model.generate(
      input_ids=input_ids,
      max_length=200,
      temperature=0.9,
      top_p=0.7,
      eos_token_id=tokenizer.eos_token_id,
  )

  decoded = tokenizer.batch_decode(out)
  assistant_message = (
      decoded[0].split("<|assistant|>\n")[-1].replace(eos_token, "")
  )

  return(assistant_message)


codeChain = prompt| mamba_coder


### mamba-chat

device = "cuda"
model_name = "havenhq/mamba-chat"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.eos_token = "<|endoftext|>"
tokenizer.pad_token = tokenizer.eos_token
tokenizer.chat_template = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta").chat_template

mamba_chat_model = MambaLMHeadModel.from_pretrained(model_name, device="cuda", dtype=torch.float16)

def mamba_chat(prompt):
  #print("mamba-chat start==========")
  messages = []
  prompt = prompt.to_string()
  messages.append(dict(role="user", content=prompt))

  input_ids = tokenizer.apply_chat_template(
              messages, return_tensors="pt", add_generation_prompt=True
  ).to(device)

  out = mamba_chat_model.generate(
      input_ids=input_ids,
      max_length=200,
      temperature=0.9,
      top_p=0.7,
      eos_token_id=tokenizer.eos_token_id,
  )

  decoded = tokenizer.batch_decode(out)
  assistant_message = (
      decoded[0].split("<|assistant|>\n")[-1].replace(eos_token, "")
  )

  return(assistant_message)

chatChain = prompt| mamba_chat

## Routing

# place holder function
def router(question):
  if ('script' in question) or ('code' in question):
    return 'code'
  else:
    return 'others'
  

def invokeModel(info):
    if info["topic"] == None:
      return chatChain
    elif "code" == info["topic"].lower():
        return codeChain
    else:
        return chatChain


full_chain = {"topic": router, "question":  RunnablePassthrough()} | RunnableLambda(invokeModel)

# ans = full_chain.batch(["Write a bash script to remove .tmp files", "What is photosynthesis?"])
# print(ans)
  
