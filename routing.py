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
import openai

def main(args):
    # Using LLM for routing name 
    if args.gpt:
        os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API Key: ")
        openai.api_key = os.environ["OPENAI_API_KEY"]
        categories = {
            "nq_open": "wikipedia based knowledge/ common sense",
            "GSM8K": "math problem, common sense",
            "MedQUAD": "medical problem, common sense",
            "code2text": "code understanding, summarization",
            "dialog_summary": "dialog summarization",
            "cnn_news": "news summarization",
            "triviaqa": "wikipedia reading and information extraction, Context-Based QA",
            "squad": "general retrieval task, Context-Based QA",
            "swde": "table based information extraction, Context-Based QA",
            "drop": "reading comprehension (requirece discrete reasoning over paragraphs), Context-Based QA"
        }
        response = openai.Completion.create(
            engine="gpt-4",
            prompt=f"Categorize the following prompt into one of the categories: {', '.join(categories.keys())}. Prompt: {args.prompt}",
            max_tokens=50,
            n=1,
            stop=None,
            temperature=0
        )
    
        # Extract the category from the response
        category = response.choices[0].text.strip() 
        model_name = parse(category)   
    # semantic routing
    else:
        routes =[]
        commen_sense_tasks = ["nq_open", "GSM8K", "MedQUAD"]
        summarization_tasks = ["code2text", "dialog_summary", "cnn_news"]
        context_tasks = ["triviaqa", "squad", "swde", "drop"]
        tasks = commen_sense_tasks + summarization_tasks + context_tasks
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


    print("Prompt: {}\r\nModel: {}".format(args.prompt, model_name))
    
def parse(category):
    model_names =["nq_open", "GSM8K", "MedQUAD", "code2text", "dialog_summary", "cnn_news", "triviaqa", "squad", "swde", "drop"]
    for model_name in model_names:
        if model_name in category.lower():
            return model_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User Defined Router')
    parser.add_argument('--limit', type=int, default=15, help='Utterance Number')
    parser.add_argument('--encoder', type=str, default="huggingface", help='Model Name')
    parser.add_argument('--optimize', action='store_true', help='Flag to trigger optimization')
    parser.add_argument('--prompt', type=str, help='A prompt string')
    parser.add_argument('--gpt', action='store_true', help='Using GPT for routing')


    args = parser.parse_args()

    main(args)