import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

import os
import argparse
from getpass import getpass
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation
from semantic_router import Route
from semantic_router.layer import RouteLayer
from semantic_router.encoders import HuggingFaceEncoder, CohereEncoder, OpenAIEncoder
import openai

from functools import partial

class ComposedModel:
    def __init__(self, device = "cuda"):
        if device == "cuda":
            torch.cuda.empty_cache()
        self.chains = {}
        self.device = device
        self.routes = []
        self.encoder = HuggingFaceEncoder()
        self.rl = RouteLayer(encoder=self.encoder, routes=self.routes)
        self.default_chain = self.create_default_chain("state-spaces/mamba-1.4b")
        self.composed_chain = None

    def create_default_chain(self, model_name):
        prompt = self.create_prompt()
        tokenizer = self.load_tokenizer(model_name)
        model = self.load_model(model_name)
        default_chain = self.create_chain(prompt, tokenizer, model)
        return default_chain
    
    ############# Models

    def create_prompt(self, in_context_prompt = ""):
        template = """{context} {question}"""
        prompt = PromptTemplate.from_template(template)
        prompt = prompt.partial(context=in_context_prompt)
        return prompt
    
    def load_tokenizer(self, model_name):
        if model_name.find("state-spaces/mamba") != -1:
            tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        return tokenizer
    
    def load_model(self, model_name, checkpoint = None):
        model = MambaLMHeadModel.from_pretrained(model_name, device=self.device, dtype=torch.float16)
        print(">>>>>>>>>>>>>> loaded pretrained from", model_name)

        if (checkpoint != None):
            model.load_state_dict(torch.load(f'{checkpoint}/pytorch_model.bin'))
        print(">>>>>>>>>>>>>> loaded checkpoint from", checkpoint)

        return model

    def create_chain(self, prompt, tokenizer, model):
        def model_generate(tokenizer, model, prompt):
            prompt = prompt.to_string()

            inputs = tokenizer.encode_plus(prompt, padding = True, return_tensors='pt')
            inputs = inputs.to(self.device)
            input_ids = inputs.input_ids

            out = model.generate(
                input_ids=input_ids,
                max_length=200,
                temperature=0.9,
                top_p=0.7,
                eos_token_id=tokenizer.eos_token_id,
            )

            input_length = input_ids.size(1)
            generated_text = tokenizer.decode(out[0][input_length:], skip_special_tokens=True) 

            return(generated_text)

        model_func = partial(model_generate, tokenizer, model)

        chain = prompt | model_func
        return chain


    ############# Router
    
    def create_route(self, task, embedding_limit = 15, limit = 45):
        commen_sense_tasks = ["nq_open", "GSM8K", "MedQUAD"]
        summarization_tasks = ["code2text", "dialog_summary", "cnn_news"]
        context_tasks = ["triviaqa", "squad", "swde", "drop"]
        
        if task in commen_sense_tasks:
            prompts, labels, metrics = load_commonsense_evaluation(task, limit = limit)
        elif task in summarization_tasks:
            prompts, labels, metrics = load_summarization_evaluation(task, limit = limit)
        elif task in context_tasks:
            prompts, labels, metrics = load_extraction_evaluation(task, limit = limit)
        route = Route(
                name=task,
                utterances = prompts[:embedding_limit],
            )
        return route
    
    def register_route_to_model(self, task, route, chain):
        self.routes.append(route)
        self.chains[task] = chain
        print(">>>>>>>>>>>>>> registered model for task", task)

    # public, optional
    def set_encoder(self, encoder_name):
        if encoder_name == "huggingface":
            self.encoder = HuggingFaceEncoder()
        elif encoder_name == "cohere":
            os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass("Enter Cohere API Key: ")
            self.encoder = CohereEncoder()
        elif encoder_name == "openai":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API Key: ")
            self.encoder = OpenAIEncoder()
    
    def router(self,question):
        return self.rl(question).name
  
    def invokeModel(self, info):
        decision = info["decision"]
        if decision == None:
            print("routed to default")
            return self.default_chain
        elif decision in self.chains:
            print("routed to chain for", decision)
            return self.chains[decision]
        else:
            print("!!! Chain not registered for", decision)
            return self.default_chain
        

    ######### Public functions
        
    ## public, necessary
    def add_model(self, task, model_name, in_context_prompt = "", checkpoint = None):
        prompt = self.create_prompt(in_context_prompt)
        tokenizer = self.load_tokenizer(model_name)
        model = self.load_model(model_name, checkpoint)
        chain = self.create_chain(prompt, tokenizer, model)
        route = self.create_route(task)
        self.register_route_to_model(task, route, chain)
        
    ## public, necessary
    def finalize_model(self):
        self.rl = RouteLayer(encoder=self.encoder, routes=self.routes)
        self.composed_chain = {"decision": self.router, "question":  RunnablePassthrough()} | RunnableLambda(self.invokeModel)

    ## public
    def batch(self, batched_prompts):
        return self.composed_chain.batch(batched_prompts)

