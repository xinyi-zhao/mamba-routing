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

commen_sense_tasks = ["nq_open", "GSM8K", "MedQUAD"]
summarization_tasks = ["code2text", "dialog_summary", "cnn_news"]
context_tasks = ["triviaqa", "squad", "swde", "drop"]
tasks = commen_sense_tasks + summarization_tasks + context_tasks

def main(args):
    # Using LLM for routing name 
    model = args.model
    if model == "gpt":
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
        categories = {
            "nq_open": "Knowledge and common sense derived from Wikipedia.",
            "GSM8K": "Solving math problems requiring common sense reasoning.",
            "MedQUAD": "Addressing medical queries with common sense reasoning.",
            "code2text": "Understanding and summarizing programming code.",
            "dialog_summary": "Summarization of conversational dialogs.",
            "cnn_news": "Summarizing news articles from CNN.",
            "triviaqa": "Extracting information from Wikipedia for context-based question answering.",
            "squad": "Retrieval tasks for general knowledge, context-based question answering.",
            "swde": "Extracting information from tables for context-based question answering.",
            "drop": "Advanced reading comprehension that involves discrete reasoning over text paragraphs for context-based question answering."
        }

        response = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=f"Categorize the following prompt into one of the categories: {', '.join(categories.keys())}. Prompt: {args.prompt}",
            max_tokens = 50
        )
    
        # Extract the category from the response
        category = response.choices[0].text.strip() 
        model_name = parse(category)   
    # semantic routing
    elif model == "semantic":
        routes =[]
        data = []
        n = args.limit
        limit = args.limit * 3 if args.optimize else args.limit

        for task in tasks:
            if task in commen_sense_tasks:
                prompts, labels, metrics = load_commonsense_evaluation(task, limit = limit)
            elif task in summarization_tasks:
                prompts, labels, metrics = load_summarization_evaluation(task, limit = limit)
            elif task in context_tasks:
                prompts, labels, metrics = load_extraction_evaluation(task, limit = limit)
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
    elif model == "vector":
        sentences = [args.prompt]
        embeddings = get_embeddings(sentences, args.limit)
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
        index = calculate_max_similarities(embeddings, args.limit)
        model_name = model_index[index]
    else:
        print(" ===== Wrong Model Name ===== ")

    print("Prompt: {}\r\nModel: {}".format(args.prompt, model_name))
    
def parse(category):
    model_names =["nq_open", "GSM8K", "MedQUAD", "code2text", "dialog_summary", "cnn_news", "triviaqa", "squad", "swde", "drop"]
    for model_name in model_names:
        if model_name in category.lower():
            return model_name
    # set default to None
    return None


def get_embeddings(sentences, limit = 2):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')

    # add utterences into sentences
    for task in tasks:
        if task in commen_sense_tasks:
            prompts, labels, metrics = load_commonsense_evaluation(task, limit = limit)
        elif task in summarization_tasks:
            prompts, labels, metrics = load_summarization_evaluation(task, limit = limit)
        elif task in context_tasks:
            prompts, labels, metrics = load_extraction_evaluation(task, limit = limit)
        sentences.extend(prompts)
    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

def calculate_max_similarities(embeddings, n=2) -> int:
    tensor_detached = embeddings.detach()
    detached_embeddings = tensor_detached.numpy()
    prompt = detached_embeddings[0]
    simiarities = [cosine_similarity(prompt.reshape(1, -1), detached_embeddings[i].reshape(1, -1))[0][0] for i in range(1, len(embeddings))]

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
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User Defined Router')
    parser.add_argument('--limit', type=int, default=15, help='Utterance Number')
    parser.add_argument('--encoder', type=str, default="huggingface", help='Model Name')
    parser.add_argument('--optimize', action='store_true', help='Flag to trigger optimization')
    parser.add_argument('--prompt', type=str, help='A prompt string')
    parser.add_argument('--model', type=str, help='Using semantic routing, or gpt or vector similarities')

    args = parser.parse_args()

    main(args)