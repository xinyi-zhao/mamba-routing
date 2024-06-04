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
import time

TEST = 80

commen_sense_tasks = ["nq_open", "GSM8K", "MedQUAD"]
summarization_tasks = ["code2text", "dialog_summary", "cnn_news"]
context_tasks = ["triviaqa", "squad", "swde", "drop"]
tasks = commen_sense_tasks + summarization_tasks + context_tasks

TOGETHER_API_KEY = "77e2c2cf255a71c3c12c6010cb94809f705dd6321b6526a08f389c63530b60bb"
def together_call(prompt,  max_tokens = 1024):
    local_model = "meta-llama/Llama-3-70b-chat-hf"
    client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz',

    )
    messages = [
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
        {"role": "user", "content": prompt}

    ]
    chat_completion = client.chat.completions.create(messages=messages,
                                                    model=local_model,
                                                    max_tokens=max_tokens,
                                                    #response_format={ "type": "json_object" },
                                                    stream=False)
    response = chat_completion.choices[0].message.content
    return response

def one_single_message(prompt,  max_tokens = 1024):
    local_model = "meta-llama/Llama-3-70b-chat-hf"
    client = OpenAI(
        api_key=TOGETHER_API_KEY,
        base_url='https://api.together.xyz',

    )
    messages = [
        {"role": "user", "content": prompt},
    ]
    chat_completion = client.chat.completions.create(messages=messages,
                                                    model=local_model,
                                                    max_tokens=max_tokens,
                                                    #response_format={ "type": "json_object" },
                                                    stream=False)
    response = chat_completion.choices[0].message.content
    return response


def main(args):
    # Using LLM for routing name 
    model = args.model
    start_time = time.time()
    s=time.gmtime(start_time)
    print("Start time is {}".format(time.strftime("%Y-%m-%d %H:%M:%S", s)))
    if model == "gpt":
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        ) 
        # categories = {
        #     "nq_open": nq_open_descp,
        #     "GSM8K": GSM8K_descp,
        #     "MedQUAD": MedQUAD_descp,
        #     "code2text": code2text_descp,
        #     "dialog_summary": dialog_summary_descp,
        #     "cnn_news": cnn_news_descp,
        #     "triviaqa": triviaqa_descp,
        #     "squad": squad_descp,
        #     "swde": swde_descp,
        #     "drop": drop_descp
        # }
        # separator = '\r\n'
        # categories_formatted = separator.join(f'{key}: {categories[key]}' for key in categories)

        test_list = []
        tokenizer = load_tokenizer()
        for task in tasks:
            if task in commen_sense_tasks:
                prompts, _, _ = load_commonsense_evaluation(task, limit = TEST, tokenizer = tokenizer)
            elif task in summarization_tasks:
                prompts, _, _ = load_summarization_evaluation(task, limit = TEST, tokenizer = tokenizer)
            elif task in context_tasks:
                prompts, _, _ = load_extraction_evaluation(task, limit = TEST, tokenizer = tokenizer)
            # print(test, len(prompts))
            test_list += [(prompts[i], task) for i in range(TEST)]
        accuracy = 0
        cnt = 0
        
        for test_case, task_name in test_list:
            cnt += 1
            if cnt % 10 ==0:
                print(cnt)
            user_msg = prompt_prefix_1 + test_case
            response  = one_single_message(user_msg, 1024)
            # # Extract the category from the response
            category = response.strip() 
            model_name = parse(category) 
            # print(model_name, task_name, test_case)
            if model_name == task_name:
                accuracy += 1  
            else:
                print(model_name, task_name, cnt)
        accuracy /= len(test_list)
    # semantic routing
    elif model == "semantic":
        encoder_name = args.encoder
        print("Encoder is {}".format(encoder_name))
        if encoder_name == "huggingface":
            print("huggingface")
            encoder = HuggingFaceEncoder()
        elif encoder_name == "cohere":
            print("cohere")
            os.environ["COHERE_API_KEY"] = os.getenv("COHERE_API_KEY") or getpass("Enter Cohere API Key: ")
            encoder = CohereEncoder()
        elif encoder_name == "openai":
            print("openai")
            os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY") or getpass("Enter OpenAI API Key: ")
            encoder = OpenAIEncoder()
        else:
            print("Wrong Encoder Name")
            return 
        
        tokenizer = load_tokenizer()
        routes, data, test_set = [],[],[]
        n = args.limit
        limit = args.limit * 2 if args.optimize else args.limit
        # test set for each subtask
        total = limit + TEST
        for task in tasks:
            if task in commen_sense_tasks:
                prompts, _, _ = load_commonsense_evaluation(task, limit = total, tokenizer = tokenizer)
            elif task in summarization_tasks:
                prompts, _, _ = load_summarization_evaluation(task, limit = total, tokenizer = tokenizer)
            elif task in context_tasks:
                prompts, _, _ = load_extraction_evaluation(task, limit = total, tokenizer = tokenizer)
            # print(task, len(prompts), n, limit, total)
            # print(prompts)
            test_set += [(prompts[i], task) for i in range(limit, total)]
            route = Route(
                name=task,
                utterances = prompts[:n],
            )
            # print(task, prompts[:n])
            routes.append(route)   
            if args.optimize:
                data += [(prompt.replace("Answer:", ""), task) for prompt in prompts[n:limit]]
                      

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
        
        accuracy = 0 
        cnt = 1
        print("The length of test set is {}".format(len(test_set)))
        for test_case, subtask in test_set:
            if cnt % 10 == 0:
                print(cnt)
            actual_subtask = rl(test_case).name
            if actual_subtask == subtask:
                accuracy += 1
            else:
                print(actual_subtask, subtask, cnt)
            cnt += 1
        accuracy = accuracy/len(test_set) 
                
    # vector similarities
    elif model == "vector":
        embeddings_load = get_embeddings_load([], args.limit)
        embeddings_prompt = get_embeddings_prompt([], TEST)
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
        accuracy = 0
        embeddings_load = embeddings_load.detach()
        detached_embeddings_load = embeddings_load.numpy()

        print(len(detached_embeddings_load), len(embeddings_prompt))
        cnt = 1
        for i in range(len(embeddings_prompt)): 
            if cnt % 10 == 0:
                print(cnt)
            cnt += 1
            prompt_embedding = get_embedding_for_one_sentence(embeddings_prompt[i]).detach().numpy()
            index = calculate_max_similarities(prompt_embedding, detached_embeddings_load)
            model_name = model_index[index]
            predicate_name = model_index[i//TEST+1]
            if model_name == predicate_name:
                accuracy += 1
            else:
                print(model_name, predicate_name)

        accuracy /= (TEST*10) 

    # in context learning 
    elif model == "ICL": 
        client = OpenAI(
            # This is the default and can be omitted
            api_key=os.environ.get("OPENAI_API_KEY"),
        )    
        test_list = []
        tokenizer = load_tokenizer()
        for task in tasks:
            if task in commen_sense_tasks:
                prompts, _, _ = load_commonsense_evaluation(task, limit = TEST, tokenizer = tokenizer)
            elif task in summarization_tasks:
                prompts, _, _ = load_summarization_evaluation(task, limit = TEST, tokenizer = tokenizer)
            elif task in context_tasks:
                prompts, _, _ = load_extraction_evaluation(task, limit = TEST, tokenizer = tokenizer)
            # print(test, len(prompts))
            test_list += [(prompts[i], task) for i in range(TEST)]
        accuracy = 0
        # print(f"Categorize the following prompt into one of the categories: \r\n{categories_formatted}. Prompt: {test_list[0][0]}")
        cnt = 0
        for test_case, task_name in test_list:
            cnt += 1
            if cnt % 10 ==0:
                print(cnt)
            # response = client.chat.completions.create(
            #     model="gpt-4o",
            #     messages=[
            #         {"role": "system", "content": parsed_system_msg},
            #         {"role": "user", "content": prompt_prefix + nq_open_msg},
            #         {"role": "assistant", "content": "nq_open"},
            #         {"role": "user", "content": prompt_prefix + GSM8K_msg},
            #         {"role": "assistant", "content": "GSM8K"},
            #         {"role": "user", "content": prompt_prefix + MedQUAD_msg},
            #         {"role": "assistant", "content": "MedQUAD"},
            #         {"role": "user", "content": prompt_prefix + code2text_msg},
            #         {"role": "assistant", "content": "code2text"},
            #         {"role": "user", "content": prompt_prefix + dialog_summary_msg},
            #         {"role": "assistant", "content": "dialog_summary"},
            #         {"role": "user", "content": prompt_prefix + cnn_news_msg},
            #         {"role": "assistant", "content": "cnn_news"},
            #         {"role": "user", "content": prompt_prefix + triviaqa_msg},
            #         {"role": "assistant", "content": "triviaqa"},
            #         {"role": "user", "content": prompt_prefix + squad_msg},
            #         {"role": "assistant", "content": "squad"},
            #         {"role": "user", "content": prompt_prefix + swde_msg},
            #         {"role": "assistant", "content": "swde"},
            #         {"role": "user", "content": prompt_prefix + drop_msg},
            #         {"role": "assistant", "content": "drop"},
            #         {"role": "user", "content": args.prompt}
            #     ]
            # )
            # print(test_case)
            response = together_call(test_case, 1024)
        
            # Extract the category from the response
            category = response.strip() 
            model_name = parse(category) 
            if model_name == task_name:
                accuracy += 1
            else:
                print(model_name, task_name, cnt)
        accuracy /= len(test_list) 

    else:
        print(" ===== Wrong Model Name ===== ")
    end_time = time.time()
    s=time.gmtime(start_time)
    print("End time is {}".format(time.strftime("%Y-%m-%d %H:%M:%S", s)))
    print(" ====================== Results ======================")
    print("Accuracy of Model {} is: {:.2%}".format(args.model, accuracy))
    print("Total execution time for model {} --- {} seconds ---".format(args.model, end_time - start_time))


def parse(category):
    model_names =["nq_open", "GSM8K", "MedQUAD", "code2text", "dialog_summary", "cnn_news", "triviaqa", "squad", "swde", "drop"]
    for model_name in model_names:
        if model_name.lower() in category.lower():
            return model_name
    # set default to None
    return None

def get_embeddings_prompt(sentences, limit = 80):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    # add utterences into sentences
    for task in tasks:
        if task in commen_sense_tasks:
            prompts, _, _ = load_commonsense_evaluation(task, limit = limit , tokenizer=tokenizer)
        elif task in summarization_tasks:
            prompts,  _, _ = load_summarization_evaluation(task, limit = limit , tokenizer=tokenizer)
        elif task in context_tasks:
            prompts,  _, _ = load_extraction_evaluation(task, limit = limit , tokenizer=tokenizer)
        sentences.extend(prompts)
    return sentences

def get_embedding_for_one_sentence(sentence):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    inputs = tokenizer([sentence], padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings[0]

def get_embeddings_load(sentences, limit = 2):
    tokenizer = AutoTokenizer.from_pretrained('facebook/contriever')
    model = AutoModel.from_pretrained('facebook/contriever')
    tokenizer2 = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    # add utterences into sentences
    for task in tasks:
        if task in commen_sense_tasks:
            prompts, _, _ = load_commonsense_evaluation(task, limit = limit , tokenizer=tokenizer2)
        elif task in summarization_tasks:
            prompts,  _, _ = load_summarization_evaluation(task, limit = limit , tokenizer=tokenizer2)
        elif task in context_tasks:
            prompts,  _, _ = load_extraction_evaluation(task, limit = limit , tokenizer=tokenizer2)
        sentences.extend(prompts)
    # Apply tokenizer
    inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    outputs = model(**inputs)

    # Mean pooling
    embeddings = mean_pooling(outputs[0], inputs['attention_mask'])
    return embeddings

def calculate_max_similarities(embedding, embeddings) -> int:
    simiarities = [cosine_similarity(embedding.reshape(1, -1), embeddings[j].reshape(1, -1))[0][0] for j in range(len(embeddings))]

    # Group simiarities by k elements and sum each group
    group_simiarities = []
    for i in range(0, len(simiarities), 3):
        group_simiarity = sum(simiarities[i:i+3])
        group_simiarities.append((i//3+1, group_simiarity))

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='User Defined Router')
    parser.add_argument('--limit', type=int, default=15, help='Utterance Number')
    parser.add_argument('--encoder', type=str, default="huggingface", help='Encoder Name')
    parser.add_argument('--optimize', action='store_true', help='Flag to trigger optimization')
    parser.add_argument('--prompt', type=str, help='A prompt string')
    parser.add_argument('--model', type=str, help='Using semantic routing, gpt, vector similarities or ICL')
    parser.add_argument('--tokenizer', type=str, help='tokenizer')

    args = parser.parse_args()

    main(args)