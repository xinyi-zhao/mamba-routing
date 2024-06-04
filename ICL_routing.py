nq_open_descp = "The nq_open category includes prompts that can be answered using a broad base of general knowledge and common sense. Prompts are derived from a variety of sources, primarily focusing on content similar to that found in Wikipedia."
GSM8K_descp = "The GSM8K category focuses on grade school mathematics problems, requiring both arithmetic skills and common sense reasoning to solve. It includes a variety of math problems typically encountered from elementary to middle school levels. These problems often involve basic operations, fractions, percentages, and simple geometry."
MedQUAD_descp ="This category is tailored for applications requiring specialized medical knowledge to answer questions. It comprises queries related to a wide range of medical topics including symptoms, diseases, treatments, and preventative healthcare."
code2text_descp = "This category is basically for understanding and summarizing code. It consists of various programming snippets, primarily in languages like Python and Java, along with explanations of what the code does."
dialog_summary_descp = "This category focuses on the task of summarizing conversations. It includes transcripts of dialogues from various contexts such as customer service calls, meetings, and casual conversations. "
cnn_news_descp = "This is tailored for the task of news article summarization. It comprises a collection of news articles from CNN, paired with concise summaries."
triviaqa_descp ="This includes trivia question that answering it require information retrieval and comprehension skills. It consists of trivia questions paired with evidence documents, requiring models to extract relevant information to formulate correct answers. The questions cover a wide range of topics including history, literature, science, and popular culture, often requiring detailed knowledge and contextual understanding."
squad_descp = "This is short for Stanford Question Answering Dataset. It is a benchmark for evaluating machine comprehension, particularly in the context of natural language understanding. It consists of questions posed by crowdworkers on a set of Wikipedia articles, where the answer to every question is a segment of text, or a 'span', from the corresponding reading passage."
swde_descp = "shorts for Structured Web Data Extraction. Prompts are usually asking for extracting structured information from unstructured web data. It includes web pages from various domains, such as automotive, real estate, and job postings, where the task is to identify and extract specific data points like prices, features, and contact information."
drop_descp = "stands for Discrete Reasoning Over Paragraphs. This category is based on natural language understanding systems to perform complex reasoning over textual data. It includes paragraphs from Wikipedia articles followed by questions that require numerical reasoning, comparison, and inference to answer. These tasks often involve extracting dates, performing arithmetic operations, or synthesizing information from multiple sentences to arrive at a conclusion."

system_message = """You will be provided with a prompt, and your task is to classify the prompt into the following categories. Choose ONLY from the list of categories provided here:
nq_open: {}
GSM8K: {}
MedQUAD: {}
code2text: {}
dialog_summary: {}
cnn_news: {}
triviaqa: {}
squad: {}
swde: {}
drop: {}
    """
nq_open_msg = "which is the state flower of himachal pradesh"
GSM8K_msg = "In a conference room, 40 chairs with a capacity of 2 people each were arranged in rows in preparation for the board meeting of a company, whose number of members was the same as the chairs' capacity. If 2/5 of the chairs were not occupied, and the rest each had two people, calculate the number of board members who did attend the meeting."
MedQUAD_msg = "How should AbobotulinumtoxinA Injection be used and what is the dosage ?"
code2text_msg = "function createInstance(defaultConfig) {\n  var context = new Axios(defaultConfig);\n  var instance = bind(Axios.prototype.request, context);\n\n  // Copy axios.prototype to instance\n  utils.extend(instance, Axios.prototype, context);\n\n  // Copy context to instance\n  utils.extend(instance, context);\n\n  return instance;\n}"+" What is the funtion of this code?"
dialog_summary_msg = "#Person1#: Mister Ewing said we should show up at the conference center at 4 o'clock, right? #Person2#: Yes, he specially asked us not to be late. Some of the people from our East York branch office are coming and he wants to make a good impression on them. How are you getting there? #Person1#: I was thinking of taking my car but I think I'm just going to take the underground because there is construction on the highway. What about you? #Person2#: I'll be taking the underground as well. Why don't we go together? I've been to the conference center only once, and I'm not sure if I can find my way around there."+" What is the summarization of this dialogue?"
cnn_news_msg = "(CNN) -- With his hands and feet shackled and his face obscured by his long hair, A former girlfriend of Stiles' said that, before the arrest, she lived in fear after going to police to identify the suspect after seeing enhanced photos from the videotape on the local news. \"I've had my share of nightmares,\" Elaine Thomas told CNN's Nancy Grace. Thomas said she screamed when she recognized the photos on television and had no choice but to contact police about the man she had thought was a \"weapons enthusiast\" with only a minor criminal record. Watch Thomas say how she felt when she saw the photos » . \"How could I not tell them who that man was? That little girl suffered unimaginable things, and I knew  apartment where her son and daughter lived. Also living in the apartment were a family friend and her daughter, the alleged assault victim. Todd Allen, Tina Allen's son, said he recognized his old apartment from scenes in the video. E-mail to a friend . CNN's Ed Payne and Ted Rowlands contributed to this report." + "What is the highlight of this article?"
triviaqa_msg = "All of Notre Dame's undergraduate students are a part of one of the five undergraduate colleges at the school or are in the First Year of Studies program. The First Year of Studies program was established in 1962 to guide incoming freshmen in their first year at the school before they have declared a major. Each student is given an academic advisor from the program who helps them to choose classes that give them exposure to any major in which they are interested. The program also includes a Learning Resource Center which provides time management, collaborative learning, and subject tutoring. This program has been recognized previously, by U.S. News & World Report, as outstanding. "+ "What entity provides help with the management of time for new students at Notre Dame?"
squad_msg = "[DOC] [TLE] MordantA mordant is a substance used to set dyes  The substrate is treated with the mordant and then dyed. [PAR] * Meta-mordanting (metachrome): The mordant is added in the dye bath itself. [PAR] * Post-mordanting (afterchrome): The dyed material is treated with a mordant." + " Mordant is the general term for a chemical which allows what to work properly?"
swde_msg = "Junior Bonner (1972) Movie Blogor IHM Blog BrowseTop 100, Trailers ...  is at his \"rugged best\"» View All Tips 2010 New Wavefront, LLC. All right, Mr. DeMille, I'm ready for my close-up.Gloria Swanson (Norma Desmond) Sunset Blvd. Summary of information above... rank: 4 language: English written by: Jeb Rosebrook directed by:"
drop_msg = "In week 11, the Lions hosted the Carolina Panthers.  A few minutes later, Detroit retook the lead when Brandon Pettigrew caught a 7-yard pass for a TD. The Lions capped their victory with Kevin Smith's third touchdown of the game, this one on a 19-yard rush. By winning this game after trailing by 17 points, the Lions are the first team in NFL history that have won 3 games in one season in which they trailed by at least 17 points. "+"How many touchdowns did Steve Smith score in the first quarter?"


prompt_prefix= "What category does the following prompt belong to? Prompt: "


prompt_prefix_1 = f"""
You will be provided with a prompt, and your task is to classify the prompt into the following categories. Choose ONLY from the list of categories provided here: 
"nq_open": description: {nq_open_descp}.
"GSM8K": description: {GSM8K_descp}. 
"MedQUAD": description: {MedQUAD_descp}. 
"code2text": description: {code2text_descp}. 
"dialog_summary": description: {dialog_summary_descp}. 
"cnn_news": description: {cnn_news_descp}.
"triviaqa": description: {triviaqa_descp}. 
"squad": description: {squad_descp}. 
"swde": description: {swde_descp}. 
"drop": description: {drop_descp}. 

Here are some of the examples.
"nq_open":  {nq_open_msg},
"GSM8K":  {GSM8K_msg},
"MedQUAD":  {MedQUAD_msg},
"code2text": {code2text_msg},
"dialog_summary": {dialog_summary_msg},
"cnn_news":  {cnn_news_msg},
"triviaqa": {triviaqa_msg},
"squad": {squad_msg},
"swde":  {swde_msg},
"drop": {drop_msg}

Which category does this prompt belong to? Prompt:
"""



parsed_system_msg = system_message.format(nq_open_descp, GSM8K_descp, MedQUAD_descp, code2text_descp, dialog_summary_descp, cnn_news_descp, triviaqa_descp, squad_descp, swde_descp, drop_descp)
