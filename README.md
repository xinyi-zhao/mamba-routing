torch version 2.2.0

# Evaluation
## Tasks
Three types of tasks:
### Common Knowledge QA: 
- nq_open: wikipedia based knowledge
- GSM8K: math problem
- MedQUAD: medical problem 
### Summarization QA:
- code2text: code understanding
- dialog_summary: dialog summarization
- cnn_news: news summarization
### Context-Based QA:
- triviaqa: wikipedia reading and information extraction
- squad: general retrieval task
- swde: table based information extraction
- drop: reading comprehension (requirece discrete reasoing over paragraphs)

```console
python evaluation.py --datasets nq_open GSM8K MedQUAD --limit 100
python evaluation.py --datasets code2text dialog_summary cnn_news  --limit 1000
python evaluation.py --datasets triviaqa squad swde drop  --limit 100
```

## Models
We support 4 model architectures:
- state-spaces/mamba-1.4b
- EleutherAI/gpt-neo-1.3B
- EleutherAI/pythia-1.4b
- facebook/opt-1.3b

```console
python evaluation.py --model EleutherAI/gpt-neo-1.3B --datasets nq_open --limit 100
```
## Performance metrics
(Reported per dataset of evaluation)
- latency: time taken across ```model.generate```
- number of input tokens
- number of output tokens
- time per output token: latency / number of output tokens

## Running with finetuned model:
```console
python evaluation.py --model state-spaces/mamba-370m --checkpoint saved_models/state-spaces_mamba-370m_nq_open/checkpoint-200 --datasets nq_open --limit 100
```

## Running with finetuned model:
python evaluation.py --model state-spaces/mamba-370m --checkpoint state-spaces_mamba-370m_nq_open/checkpoint-500 --batch_size 32 --datasets nq_open --limit 100

## Todo
- [ ] Run evaluation on across all models
- [ ] Be careful about the prompt design. 

# Finetune
Same datasets and tasks
```console
python finetune.py --dataset nq_open  --limit 300 --num_epochs 3
```

## Performance metrics
(Logged by trainer)
- train_runtime: time taken for training
- train_samples_per_second
- train_steps_per_second
- train_loss: final loss

```console
{'train_runtime': 250.309, 'train_samples_per_second': 3.596, 'train_steps_per_second': 0.899, 'train_loss': 0.6560872395833334, 'epoch': 3.0}
```

## Todo
- [ ] Support fine-tuning for one of gpt-neo/ pythia/ opt
- [ ] For one task, compare fine-tuning on mamba vs gpt-neo/ pythia/ opt
- [ ] Finetune mamba on other tasks
- [ ] Prompt based smaller model (edit prompt template in composed.py to do in-context learning for each sub-model)
- [ ] decrease the effect of prompt design during evaluation

# Routing
- routing between three types of tasks
- routing inside each types of task(different domain)

- ```limit```: # of utterance loaded from dataset
- ```encoder``` encoder type: huggingface, openai, cohere
- ```gpt``` using LLM model or semantic routing
- ```optimze``` whether to optimize the semantic routing model
- ```prompt``` prompt
```console
python3 routing.py --limit 5 --encoder huggingface  --optimize --gpt --prompt " Still searching for their first win, the Bengals flew to Texas Stadium for a Week 5 interconference duel with the Dallas Cowboys.  In the first quarter, Cincinnati trailed early as Cowboys kicker Nick Folk got a 30-yard field goal, along with RB Felix Jones getting a 33-yard TD run.  In the second quarter, Dallas increased its lead as QB Tony Romo completed a 4-yard TD pass to TE Jason Witten.  The Bengals would end the half with kicker Shayne Graham getting a 41-yard and a 31-yard field goal. In the third quarter, Cincinnati tried to rally as QB Carson Palmer completed an 18-yard TD pass to WR T. J. Houshmandzadeh.  In the fourth quarter, the Bengals got closer as Graham got a 40-yard field goal, yet the Cowboys answered with Romo completing a 57-yard TD pass to WR Terrell Owens.  Cincinnati tried to come back as Palmer completed a 10-yard TD pass to Houshmandzadeh (with a failed 2-point conversion), but Dallas pulled away with Romo completing a 15-yard TD pass to WR Patrick Crayton.Which Bengals receiver scored two touchdowns?"
```

## Todo
- [ ] Improve accuracy
- [ ] Try writing a different similarity match

# Composed 

## Current integration (unbatched)
Prototyping, may be used for accuracy testing
- Defines a langchain for each fine-tuned model.
- Routes via semantic routing.
- Use evaluate_composed.py for evaluation.

### Usage and Evaluation

Construct model
```console
composed = ComposedModel(device)
composed.add_model("code2text", "mrm8488/mamba-coder", in_context_prompt = "", checkpoint = None)
composed.finalize_model()
generated = composed.batch(prompts)
```

Evaluation
```console
python evaluate_composed.py --datasets code2text nq_open --limit 50
```

## Composed model with parallel batched execution
Per batch 
- Route each prompt to a model (use tag for perfect routing simulation)
- Bucket prompts by their routed model
- Execute batched inference on each model in parallel

### Evaluation
- Randomize incoming order of prompts from different datasets
- Tag each prompt with its label for metric computation after
- Tag each prompt with its task for perfect routing simulation

### Todo
- [ ] Dev
- [ ] Run latency measurements

## System building and latency test
### Todo
- [ ] Write the pipeline
- [ ] Run the test