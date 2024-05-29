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

python evaluation.py --datasets nq_open GSM8K MedQUAD --limit 100

python evaluation.py --datasets code2text dialog_summary cnn_news  --limit 1000

python evaluation.py --datasets triviaqa squad swde drop  --limit 100

## Running with finetuned model:
python evaluation.py --model state-spaces/mamba-370m --checkpoint state-spaces_mamba-370m_nq_open/checkpoint-500 --batch_size 32 --datasets nq_open --limit 100

## Todo
- Ensure evaluation runs on gpt-neo, opt, pythia, hybrid h3
- Add inference latency measurement
- Be careful about the prompt design. 

# Finetune
Same datasets and tasks
python finetune.py --dataset summarization  --limit 100

### Sample code used for fine-tuning mamba-chat (/training)
~~python train_mamba.py --model state-spaces/mamba-790m --tokenizer EleutherAI/gpt-neox-20b --learning_rate 5e-5 --batch_size 1 --gradient_accumulation_steps 4 --optim paged_adamw_8bit --data_path ./ultrachat_small.jsonl --num_epochs 3~~

## Todo
- ~Load dataset (edit /training/data.py to use our dataset)~
- ~Finetune and save model (see /training)~
- Add training throughput measurement
- Finetuning all all the tasks
- Prompt based smaller model? (edit prompt template in composed.py to do in-context learning for each sub-model)
- decrease the effect of prompt design during evaluation

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
- Replace placeholder router function in composed.py
- decrease the effect of some prompt-design on our side

# Integration
- Combine the evaluation to the routing process (see evaluate_composed.py)
## Todo
- Run tests
