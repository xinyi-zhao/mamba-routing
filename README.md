torch version 2.2.0

# Evaluation
Three types of tasks:
###Common Knowledge QA: 
- nq_open: wikipedia based knowledge
- GSM8K: math problem
- MedQUAD: medical problem 
###Summarization QA:
- code2text: code understanding
- dialog_summary: dialog summarization
- cnn_news: news summarization
###Context-Based QA:
- triviaqa: wikipedia reading and information extraction
- squad: general retrieval task
- swde: table based information extraction
- drop: reading comprehension (requirece discrete reasoing over paragraphs)

python evaluation.py --datasets nq_open GSM8K MedQUAD --limit 100
python evaluation.py --datasets code2text dialog_summary cnn_news  --limit 1000
python evaluation.py --datasets triviaqa squad swde drop  --limit 100

## Todo
Be careful about the prompt design. 

# Finetune

## Todo
- load dataset
- finetune and save model
- prompt based smaller model?
- decrease the effect of prompt design during evaluation

# Routing
- routing between three types of tasks
- routing inside each types of task(different domain)
## Todo
Combine the evaluation to the routing process
decrease the effect of some prompt-design on our side