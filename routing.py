from semantic_router import Route
import os
import argparse
from getpass import getpass
from semantic_router.layer import RouteLayer
from loaddata.commonsense import load_commonsense_evaluation
from loaddata.summarization import load_summarization_evaluation
from loaddata.informationextraction import load_extraction_evaluation
from langchain_core.prompts import PromptTemplate
from semantic_router.encoders import HuggingFaceEncoder, CohereEncoder, OpenAIEncoder
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from ICL_routing import *

commen_sense_tasks = ["nq_open", "GSM8K", "MedQUAD"]
summarization_tasks = ["code2text", "dialog_summary", "cnn_news"]
context_tasks = ["triviaqa", "squad", "swde", "drop"]
TASKS = commen_sense_tasks + summarization_tasks + context_tasks

def main(args):
    # Using LLM for routing name 
    model = args.model
    if model == "gpt":
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        ) 
        categories = {
            "nq_open": nq_open_descp,
            "GSM8K": GSM8K_descp,
            "MedQUAD": MedQUAD_descp,
            "code2text": code2text_descp,
            "dialog_summary": dialog_summary_descp,
            "cnn_news": cnn_news_descp,
            "triviaqa": triviaqa_descp,
            "squad": squad_descp,
            "swde": swde_descp,
            "drop": drop_descp
        }
        separator = '\r\n'
        categories_formatted = separator.join(f'{key}: {categories[key]}' for key in categories)
        user_msg = f"Categorize the following prompt into one of the categories: {categories_formatted}. Prompt: {args.prompt}"
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role":"user", "content":user_msg}
            ],
            max_tokens = 50
        )
    
        # # Extract the category from the response
        category = response.choices[0].message.content.strip() 
        model_name = parse(category)   
    # semantic routing
    elif model == "semantic":
        tokenizer = load_tokenizer()
        routes =[]
        data = []
        n = args.limit
        limit = args.limit * 3 if args.optimize else args.limit

        for task in TASKS:
            if task in commen_sense_tasks:
                prompts, _, _ = load_commonsense_evaluation(task, limit = limit, tokenizer = tokenizer)
            elif task in summarization_tasks:
                prompts, _, _ = load_summarization_evaluation(task, limit = limit, tokenizer = tokenizer)
            elif task in context_tasks:
                prompts, _, _ = load_extraction_evaluation(task, limit = limit, tokenizer = tokenizer)
            route = Route(
                    name=task,
                    utterances = prompts[:n],
                )
        
            if args.optimize:
                data += [(prompt.replace("Answer:", ""), task) for prompt in prompts[n:limit]]
            routes.append(route)     

        encoder_name = args.encoder
        if encoder_name == "huggingface":
            encoder = HuggingFaceEncoder()
        elif encoder_name == "cohere":
            os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass("Enter Cohere API Key: ")
            encoder = CohereEncoder()
        elif encoder_name == "openai":
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API Key: ")
            encoder = OpenAIEncoder()
        

        rl = RouteLayer(encoder=encoder, routes=routes)

        if args.optimize:
            X, y = zip(*data)
            # TO DO Optimization Layer here
            print("==================== Optimize the Router ==================== ")
            route_thresholds = rl.get_thresholds()
            print("Default route thresholds:", route_thresholds)
            rl.fit(X=X, y=y)
            route_thresholds = rl.get_thresholds()
            print("Updated route thresholds:", route_thresholds)

        model_name = rl(args.prompt).name
    # vector similarities
    elif model == "vector":
        router_embeddings = get_vector_model()
        model_name = get_model_name(args.prompt, router_embeddings, limit = args.limit)

    # in context learning 
    elif model == "ICL": 
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        ) 
        response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": parsed_system_msg},
            {"role": "user", "content": prompt_prefix + nq_open_msg},
            {"role": "assistant", "content": "nq_open"},
            {"role": "user", "content": prompt_prefix + GSM8K_msg},
            {"role": "assistant", "content": "GSM8K"},
            {"role": "user", "content": prompt_prefix + MedQUAD_msg},
            {"role": "assistant", "content": "MedQUAD"},
            {"role": "user", "content": prompt_prefix + code2text_msg},
            {"role": "assistant", "content": "code2text"},
            {"role": "user", "content": prompt_prefix + dialog_summary_msg},
            {"role": "assistant", "content": "dialog_summary"},
            {"role": "user", "content": prompt_prefix + cnn_news_msg},
            {"role": "assistant", "content": "cnn_news"},
            {"role": "user", "content": prompt_prefix + triviaqa_msg},
            {"role": "assistant", "content": "triviaqa"},
            {"role": "user", "content": prompt_prefix + squad_msg},
            {"role": "assistant", "content": "squad"},
            {"role": "user", "content": prompt_prefix + swde_msg},
            {"role": "assistant", "content": "swde"},
            {"role": "user", "content": prompt_prefix + drop_msg},
            {"role": "assistant", "content": "drop"},
            {"role": "user", "content": args.prompt}
        ]
        )
    
        # Extract the category from the response
        category = response.choices[0].message.content.strip() 
        model_name = parse(category)   

    else:
        print(" ===== Wrong Model Name ===== ")

    print("Prompt: {}\r\nModel: {}".format(args.prompt, model_name))
    
def get_model_name(prompt_embedding, router_embeddings, limit):
    model_index = {
            1: "nq_open",
            2: "GSM8K",
            3: "MedQUAD",
            4: "code2text",
            5: "dialog_summary",
            6: "cnn_news",
            7: "triviaqa",
            8: "squad",
            9: "swde",
            10: "drop"
        }
    index = calculate_max_similarities(prompt_embedding, router_embeddings, limit )
    model_name = model_index[index]
    return model_name 

def get_vector_model():
    sentences = []
    embeddings = get_embeddings(sentences, args.limit)
    return embeddings
    

def parse(category):
    model_names =["nq_open", "GSM8K", "MedQUAD", "code2text", "dialog_summary", "cnn_news", "triviaqa", "squad", "swde", "drop"]
    for model_name in model_names:
        if model_name.lower() in category.lower():
            return model_name
    # set default to None
    return None


def get_embeddings(sentences, limit = 2):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # add utterences into sentences
    for task in TASKS:
        if task in commen_sense_tasks:
            prompts, _, _ = load_commonsense_evaluation(task, limit = limit, tokenizer=tokenizer2)
        elif task in summarization_tasks:
            prompts,  _, _ = load_summarization_evaluation(task, limit = limit, tokenizer=tokenizer2)
        elif task in context_tasks:
            prompts,  _, _ = load_extraction_evaluation(task, limit = limit, tokenizer=tokenizer2)
        sentences.extend(prompts)
    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

def get_embeddings_dataset(embedding_tasks):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    sentences = []
    for embedding_task in embedding_tasks:
        sentences.append(embedding_task["prompt"])
    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

def calculate_max_similarities(prompt, router_embeddings, n=2) -> int:
    prompt = get_embedding_for_one_sentence(prompt).detach().numpy()
    tensor_detached = router_embeddings.detach()
    detached_embeddings = tensor_detached.numpy()
    simiarities = [cosine_similarity(prompt.reshape(1, -1), detached_embeddings[i].reshape(1, -1))[0][0] for i in range(0, len(router_embeddings))]

    # Group simiarities by k elements and sum each group
    group_simiarities = []
    for i in range(0, len(simiarities), n):
        group_simiarity = sum(simiarities[i:i+n])
        group_simiarities.append((i//n+1, group_simiarity))

    # Find the group with the largest summed difference
    max_sim_group = max(group_simiarities, key=lambda x: x[1])

    return max_sim_group[0] 

def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings

def load_tokenizer():
    if args.tokenizer.find("state-spaces/mamba") != -1:
        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer

def get_embedding_for_one_sentence(sentence):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings[0]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User Defined Router')
    parser.add_argument('--limit', type=int, default=15, help='Utterance Number')
    parser.add_argument('--encoder', type=str, default="huggingface", help='Model Name')
    parser.add_argument('--optimize', action='store_true', help='Flag to trigger optimization')
    parser.add_argument('--prompt', type=str, help='A prompt string')
    parser.add_argument('--model', type=str, help='Using semantic routing, gpt, vector similarities or ICL')
    parser.add_argument('--tokenizer', type=str, help='tokenizer')

    args = parser.parse_args()

    main(args)