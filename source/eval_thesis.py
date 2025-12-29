# Import necessary modules
import time
import torch
import torch.nn as nn
import sys
import os

# Add the thesis directory to sys.path to allow imports from lib
thesis_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Muraro-Nicola-Master-Thesis", "wanda")
if thesis_path not in sys.path:
    sys.path.append(thesis_path)

import pdb
from lm_eval import evaluator, tasks
from lib.tasks import EvalHarnessAdaptor

# Import get_loaders function from data module within the same directory
from lib.data import get_loaders 
from datasets import load_dataset
from collections import defaultdict
import fnmatch
import evaluate
import json
from tqdm import tqdm
import math
import code_bert_score



def eval_zero_shot(model_name, model, tokenizer, task_list=["boolq","rte","hellaswag","winogrande","arc_challenge","arc_easy","openbookqa"], 
        num_fewshot=0, use_accelerate=False, add_special_tokens=False):
    def pattern_match(patterns, source_list):
        task_names = set()
        for pattern in patterns:
            for matching in fnmatch.filter(source_list, pattern):
                task_names.add(matching)
        return list(task_names)
    task_names = pattern_match(task_list, tasks.ALL_TASKS)
    model_args = f"pretrained={model_name},cache_dir=./llm_weights"
    limit = None 
    if "70b" in model_name or "65b" in model_name:
        limit = 2000
    if use_accelerate:
        model_args = f"pretrained={model_name},cache_dir=./llm_weights,use_accelerate=True"
    results = evaluator.simple_evaluate(
        model="hf-causal-experimental",
        model_args=model_args,
        tasks=task_names,
        num_fewshot=num_fewshot,
        batch_size=None,
        device=None,
        no_cache=True,
        limit=limit,
        description_dict={},
        decontamination_ngrams_path=None,
        check_integrity=False,
        pretrained_model=model,
        tokenizer=tokenizer, 
        add_special_tokens=add_special_tokens
    )

    return results 





# ---------------------------------- PPL ---------------------------------- #


# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_ppl(model, tokenizer, sqlen, device=torch.device("cuda:0"), verbose=False, dataset="wikitext2", batch_size=1):
    # Set datasetx
    #model.to(device)
    # Print status
    print(f"evaluating on {dataset}")
    
    # Get the test loader
    _, testloader, _ = get_loaders(
        dataset, seed=0, seqlen=sqlen, tokenizer=tokenizer, train=False
    )

    # Evaluate ppl in no grad context to avoid updating the model
    start_time = time.time()
    with torch.no_grad():
        ppl = eval_ppl_dataset(model, testloader, sqlen, bs=1, device=device, verbose=verbose)

    if verbose:
        print("execution time:", time.time() - start_time, flush=True)
        
    return ppl 


# Function to evaluate perplexity (ppl) specifically on the wikitext dataset
def eval_ppl_dataset(model, testenc, sqlen, bs=1, device=None, verbose=False):
    # Get input IDs
    testenc = testenc.input_ids
    
    # Calculate number of samples
    nsamples = testenc.numel() // sqlen

    # List to store negative log likelihoods
    nlls = []
    if verbose:
        print(f"nsamples {nsamples}")
    
    # Loop through each batch
    for i in range(0,nsamples,bs): #range(0,nsamples,bs): 
        if verbose and i % 50 == 0:
            print(f"sample {i}")

        # Calculate end index
        j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        inputs = testenc[:,(i * sqlen):(j * sqlen)].to(device)
        inputs = inputs.reshape(j-i, sqlen)
        
        # Forward pass through the model
        lm_logits = model(inputs).logits

        # Shift logits and labels for next token prediction
        shift_logits = lm_logits[:, :-1, :].contiguous()
        shift_labels = inputs[:, 1:]

        # Compute loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.reshape(-1))

        # Calculate negative log likelihood
        neg_log_likelihood = loss.float() * sqlen * (j-i)

        # Append to list of negative log likelihoods
        nlls.append(neg_log_likelihood)

    # Compute perplexity
    ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * sqlen))

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    return ppl.item()


# ---------------------------------- END PPL ---------------------------------- #


# ---------------------------------- ACC ---------------------------------- #


def eval_acc(args, model, tokenizer, sqlen, device=torch.device("cuda:0"), dataset="svamp", shot='few', verbose=False, batch_size=1):
    print(f'evaluating on {dataset}')

    # Get the test loader
    _, test_data, pad_token = get_loaders(
        dataset, seed=0, seqlen=sqlen, tokenizer=tokenizer,  train=False
    )
    if test_data:
        test_prompt, test_answer = test_data

    # Evaluate acc in no grad context to avoid updating the model
    start_time = time.time()
    
    with torch.no_grad():
        if dataset == "svamp":
            acc = eval_acc_svamp(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["svamp"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)
        elif dataset == "gsm8k":
            acc = eval_acc_gsm8k(model, tokenizer, sqlen, batch_size=batch_size["gsm8k"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)
        elif dataset == "mawps":
            acc = eval_acc_mawps(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["mawps"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)
        elif dataset.startswith("anli"):
            acc = eval_acc_anli(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["anli_r1"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose) #batch_size=8
        elif dataset == "esnli":
            acc = eval_acc_esnli(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["esnli"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)
        elif dataset == "rte":
            acc = eval_acc_rte(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["rte"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)
        elif dataset == "boolq":
            acc = eval_acc_boolq(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["boolq"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose) #batch_size=8
        elif dataset == "commonsense_qa":
            acc = eval_acc_commonsense_qa(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["commonsense_qa"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)
        elif dataset == "race":
            acc = eval_acc_race(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["race"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose) #batch_size=2
        elif dataset == "winogrande":
            acc = eval_acc_winogrande(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=batch_size["winogrande"], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)

    if verbose:
        print("execution time:", time.time() - start_time, flush=True)

    return acc





def eval_acc_svamp(model, test_prompt, test_answer, tokenizer, sqlen, batch_size=1, pad_token=None, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)

    # List to store negative log likelihoods
    if verbose:
        print(f"n_batch {n_batch}")

    num_correct = 0

    # Loop through each batch
    for i in range(n_batch): #nsamples
        
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))

        # Prepare inputs and move to device
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if eval_rationale:
                prompt = gsm8k_cot_template.format(prompt=test_prompt[index_element]).strip()
            else:
                prompt = gsm8k_question_answer_template.format(prompt=test_prompt[index_element]).strip()
            
            batch.append(prompt)
        

        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 
        
        pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256) #max_new_tokens=256

        
        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            try:
                output_decoded_clean = output_i.split('\n\n')[0].split('####')[-1].strip()
            except IndexError:
                output_decoded_clean = ''

            label = test_answer[i*batch_size + k]

            try:
                if eval(output_decoded_clean) == eval(label):
                    num_correct += 1
            except:
                pass
            
    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

mawps_question_answer_template = """
Question: A painter needed to paint 12 rooms in a building. Each room takes 7 hours to paint. If he already painted 5 rooms, how much longer will he take to paint the rest?	
Answer: (7.0*(12.0-5.0))

Question: Willie had 4 papaya . He carve each papaya into 6 slices . How many papaya slices did Willie make?	
Answer: 6*4

Question: For homework Nancy had 17 math problems and 15 spelling problems. If she can finish 8 problems in an hour how long will it take her to finish all the problems?	
Answer: ((17.0+15.0)/8.0)

{prompt}
"""


def eval_acc_gsm8k(model, tokenizer, sqlen, device=torch.device("cuda:0"), shot='few', pad_token=None, eval_rationale=False, verbose=False, batch_size=1):
    # load gsm8k test set
    test_dataset = load_dataset('gsm8k', 'main', split='test')

    correct = 0
    total = 0
    
    list_examples = test_dataset.select(range(len(test_dataset)))
    n_batch = math.ceil(len(list_examples)/batch_size)
    n_samples = len(test_dataset)
    
    
    for i in range(n_batch): #tqdm(test_dataset.select(range(len(test_dataset)))):
        #print(f"batch {i+1}/{n_batch}", flush=True)
        this_batch_size = min(batch_size, abs(len(list_examples) - i*batch_size))
        
        batch = []
        test_answer = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            example = list_examples[index_element]
            
            question = example['question']
            label = example['answer'].split('####')[1].strip()
            test_prompt = f'Question: {question}\nAnswer: '

            if eval_rationale:
                prompt = gsm8k_cot_template.format(prompt=test_prompt).strip()
            else:
                prompt = gsm8k_question_answer_template.format(prompt=test_prompt).strip()
            
            batch.append(prompt)
            test_answer.append(label)

        tokenized_input = tokenizer(batch, return_tensors="pt", padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 
        
        pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256) #max_new_tokens=256
        
        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            try:
                output_decoded_clean = output_i.split('\n\n')[0].split('####')[-1].strip()
            except IndexError:
                output_decoded_clean = ''
            
            label = test_answer[k]
            
            try:
                if eval(output_decoded_clean) == eval(label):
                    correct += 1
            except:
                pass

            total += 1
    
    acc = correct / total
    return acc

gsm8k_question_answer_template = """
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: 5

{prompt}
"""

gsm8k_cot_template = """
Question: Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?
Answer: Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72

Question: Weng earns $12 an hour for babysitting. Yesterday, she just did 50 minutes of babysitting. How much did she earn?
Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute. Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10. #### 10

Question: Betty is saving money for a new wallet which costs $100. Betty has only half of the money she needs. Her parents decided to give her $15 for that purpose, and her grandparents twice as much as her parents. How much more money does Betty need to buy the wallet?
Answer: In the beginning, Betty has only 100 / 2 = $<<100/2=50>>50. Betty's grandparents gave her 15 * 2 = $<<15*2=30>>30. This means, Betty needs 100 - 50 - 30 - 15 = $<<100-50-30-15=5>>5 more. #### 5

{prompt}
"""

def eval_acc_mawps(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)
    
    

    # List to store negative log likelihoods
    if verbose:
        print(f"n_batch {n_batch}")

    num_correct = 0

    # Loop through each batch
    for i in range(n_batch): #nsamples
        
        
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        # Calculate end index
        #j = min(i+bs, nsamples)
        
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if eval_rationale:
                prompt = gsm8k_cot_template.format(prompt=test_prompt[index_element]).strip()
            else:
                prompt = gsm8k_question_answer_template.format(prompt=test_prompt[index_element]).strip()
            
            batch.append(prompt)
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 

        # Forward pass through the model

        pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256) #max_new_tokens=256


        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            try:
                output_decoded_clean = output_i.split('\n\n')[0].split('####')[-1].strip()
            except IndexError:
                output_decoded_clean = ''

            label = test_answer[i*batch_size + k]

            try:
                if eval(output_decoded_clean) == eval(label):
                    num_correct += 1
            except:
                pass
        

    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

anli_premise_hypothesis_label_template = """
Premise: Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context. The earliest activities in the documentation and description of language have been attributed to the 4th century BCE Indian grammarian Pāṇini, who wrote a formal description of the Sanskrit language in his "Aṣṭādhyāyī ".
Hypothesis: Form and meaning are the only aspects of language linguistics is concerned with.	
Label: contradiction

Premise: Franco Zeffirelli, KBE Grande Ufficiale OMRI (] ; born 12 February 1923) is an Italian director and producer of operas, films and television. He is also a former senator (1994–2001) for the Italian centre-right "Forza Italia" party. Recently, Italian researchers have found that he is one of the few distant relatives of Leonardo da Vinci.
Hypothesis: Franco Zeffirelli had a political career	
Label: entailment

Premise: Eme 15 is the self-titled debut studio album by Mexican-Argentine pop band, Eme 15. The album was released in Mexico and Latin America on June 26, 2012 through Warner Music México, and features songs from the Nickelodeon Latin America and Televisa musical television series, Miss XV.	
Hypothesis: Eme 15 was released in Mexico and Latin America on June 27, 2012 through Warner Music México, and features songs from the Nickelodeon Latin America and Televisa musical television series, Miss XV.
Label: contradiction

Premise: Almost Sunrise is a 2016 American documentary film directed by Michael Collins. It recounts the story of two Iraq veterans, Tom Voss and Anthony Anderson, who, in an attempt to put their combat experience behind them, embark on a 2,700-mile trek on foot across America. It made its world premiere on the opening night of the Telluride Mountainfilm Festival on 27 May, 2016.
Hypothesis: Tom and Anthony have both killed someone.	
Label: neutral

Premise: Sergei Mikhailovich Grinkov (Russian: Серге́й Миха́йлович Гринько́в , February 4, 1967 — November 20, 1995) was a Russian pair skater. Together with partner and wife Ekaterina Gordeeva, he was the 1988 and 1994 Olympic Champion and a four-time World Champion.
Hypothesis: Sergei Mikhailovich Grinkov became the 1988 Olympic Champion with his partner while his wife cheered from the audience.
Label: contradiction

Premise: Lee Hong-gi (; ] ; Japanese:イ•ホンギ) is a South Korean singer-songwriter, actor, writer, and fashion designer. He is best known for his singing abilities and being the main singer of the South Korean rock band F.T. Island. Lee released his first solo extended play "FM302" in South Korea on 18 November 2015 and his Japanese album "AM302" on 9 December 2015.
Hypothesis: The South Korean rock band F.T. Island is best known for it's lead singer, Lee Hong-gi.
Label: neutral

Premise: Lost Moon: The Perilous Voyage of Apollo 13 (published in paperback as Apollo 13), is a non-fiction book first published in 1994 by astronaut James Lovell and journalist Jeffrey Kluger, about the failed April 1970 Apollo 13 lunar landing mission which Lovell commanded. The book is the basis of the 1995 film adaptation "Apollo 13", directed by Ron Howard.
Hypothesis: the book wouldnt have happened if we didnt try to go into space
Label: entailment

Premise: Will Wheaton, born Willie Mack Wheaton Jr. (born October 26, 1972) is an American singer, songwriter and musician. He grew up in Los Angeles and is the son of Gospel singer Juanita Wheaton. He studied music in his teens and was eventually tutored by Furman Fordham, whose former students include Lena Horne.
Hypothesis: Willie Mack Wheaton Jr. was born 5 days away from the end of the 10th month of 1972
Label: entailment

Premise: 3096 (3096 Tage) is a 2013 German drama film directed by Sherry Hormann. The film is based on the true story of Natascha Kampusch, a 10-year-old girl and her eight-year ordeal being kidnapped by Wolfgang Přiklopil. Northern Irish actress Antonia Campbell-Hughes portrays Kampusch, while Thure Lindhardt plays Přiklopil.
Hypothesis: Lindhardt wrote 3096.
Label: neutral

{prompt}
"""

anli_premise_hypothesis_rationale_label_template = """
Premise: Linguistics is the scientific study of language, and involves an analysis of language form, language meaning, and language in context. The earliest activities in the documentation and description of language have been attributed to the 4th century BCE Indian grammarian Pāṇini, who wrote a formal description of the Sanskrit language in his "Aṣṭādhyāyī ".
Hypothesis: Form and meaning are the only aspects of language linguistics is concerned with.	
Label: Linguistics involves an analysis of language form, language meaning, and language in context, so context is also a crucial aspect. The algorithm missed this point, perhaps. So the answer is contradiction.

Premise: Franco Zeffirelli, KBE Grande Ufficiale OMRI (] ; born 12 February 1923) is an Italian director and producer of operas, films and television. He is also a former senator (1994–2001) for the Italian centre-right "Forza Italia" party. Recently, Italian researchers have found that he is one of the few distant relatives of Leonardo da Vinci.
Hypothesis: Franco Zeffirelli had a political career	
Label: Franco Zeffirelli was a senator so he had a political career. The system likely was fooled because I used words not used in the context. So the answer is entailment.

Premise: Eme 15 is the self-titled debut studio album by Mexican-Argentine pop band, Eme 15. The album was released in Mexico and Latin America on June 26, 2012 through Warner Music México, and features songs from the Nickelodeon Latin America and Televisa musical television series, Miss XV.	
Hypothesis: Eme 15 was released in Mexico and Latin America on June 27, 2012 through Warner Music México, and features songs from the Nickelodeon Latin America and Televisa musical television series, Miss XV.
Label: The album was released in June 26, 2012 not June 27, 2012. I added alot of correct info but changed one small detail. So the answer is contradiction.

Premise: Almost Sunrise is a 2016 American documentary film directed by Michael Collins. It recounts the story of two Iraq veterans, Tom Voss and Anthony Anderson, who, in an attempt to put their combat experience behind them, embark on a 2,700-mile trek on foot across America. It made its world premiere on the opening night of the Telluride Mountainfilm Festival on 27 May, 2016.
Hypothesis: Tom and Anthony have both killed someone.	
Label: The prompt references combat experience, but is vague, so you don't know if that entailed killing anyone. The system might have gotten this wrong because it doesn't understand the relationship between combat in Iraq and death. So the answer is neutral.

Premise: Sergei Mikhailovich Grinkov (Russian: Серге́й Миха́йлович Гринько́в , February 4, 1967 — November 20, 1995) was a Russian pair skater. Together with partner and wife Ekaterina Gordeeva, he was the 1988 and 1994 Olympic Champion and a four-time World Champion.
Hypothesis: Sergei Mikhailovich Grinkov became the 1988 Olympic Champion with his partner while his wife cheered from the audience.
Label: It has been stated that Sergei Mikhailovich Grinkov skated and won the Olympic Championship with his partner and wife in 1988. So the answer is contradiction.

Premise: Lee Hong-gi (; ] ; Japanese:イ•ホンギ) is a South Korean singer-songwriter, actor, writer, and fashion designer. He is best known for his singing abilities and being the main singer of the South Korean rock band F.T. Island. Lee released his first solo extended play "FM302" in South Korea on 18 November 2015 and his Japanese album "AM302" on 9 December 2015.
Hypothesis: The South Korean rock band F.T. Island is best known for it's lead singer, Lee Hong-gi.
Label: Lee Hong-gi is best known for his talents, according to the context; however, it is not mentioned that his band is most known for him. All squares are rectangles, but not all rectangles are squares. Using key words from the context (Lee Hong-gi, best known, the South Korean rock band) matched and I'm assuming the computer thought the matching strings of text was a positive. So the answer is neutral.

Premise: Lost Moon: The Perilous Voyage of Apollo 13 (published in paperback as Apollo 13), is a non-fiction book first published in 1994 by astronaut James Lovell and journalist Jeffrey Kluger, about the failed April 1970 Apollo 13 lunar landing mission which Lovell commanded. The book is the basis of the 1995 film adaptation "Apollo 13", directed by Ron Howard.
Hypothesis: the book wouldnt have happened if we didnt try to go into space
Label: there wouldnt have been a failed mission if we never tried to go to space. So the answer is entailment.

Premise: Will Wheaton, born Willie Mack Wheaton Jr. (born October 26, 1972) is an American singer, songwriter and musician. He grew up in Los Angeles and is the son of Gospel singer Juanita Wheaton. He studied music in his teens and was eventually tutored by Furman Fordham, whose former students include Lena Horne.
Hypothesis: Willie Mack Wheaton Jr. was born 5 days away from the end of the 10th month of 1972
Label: Weird way to say date again. It hates it. So the answer is entailment.

Premise: 3096 (3096 Tage) is a 2013 German drama film directed by Sherry Hormann. The film is based on the true story of Natascha Kampusch, a 10-year-old girl and her eight-year ordeal being kidnapped by Wolfgang Přiklopil. Northern Irish actress Antonia Campbell-Hughes portrays Kampusch, while Thure Lindhardt plays Přiklopil.
Hypothesis: Lindhardt wrote 3096.
Label: It is unknown who wrote the film. The system was confused by the facts of the narrative. So the answer is neutral.

{prompt}
"""

def eval_acc_anli(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)
    #verbose = True

    # List to store negative log likelihoods
    if verbose:
        print(f"n_batch {n_batch}")

    num_correct = 0

    labels = ['entailment', 'neutral', 'contradiction']

    # Loop through each batch
    for i in range(n_batch): #range(nsamples)
        
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if eval_rationale:
                prompt = anli_premise_hypothesis_rationale_label_template.format(prompt=test_prompt[index_element]).strip()
            else:
                prompt = anli_premise_hypothesis_label_template.format(prompt=test_prompt[index_element]).strip()
            
            batch.append(prompt)
        
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 
        
        
        # Forward pass through the model
        if eval_rationale:
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256) #max_new_tokens=256
        else:
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20) #max_new_tokens=20


        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if eval_rationale:
                output_decoded_first_line = output_i.split('\n\n')[0].strip()
                output_decoded_last_sentence = output_decoded_first_line.split(".")[-2].strip()
                output_decoded_clean = output_decoded_last_sentence.split('So the answer is ')[-1].strip()
            else:
                output_decoded_clean = output_i.split('\n\n')[0].strip()

            label = test_answer[i*batch_size + k]

            try:
                if output_decoded_clean == label:
                    num_correct += 1
            except:
                pass



    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

esnli_premise_hypothesis_label_template = """
Premise: A person on a horse jumps over a broken down airplane.	
Hypothesis: A person is training his horse for a competition.
Label: neutral

Premise: A person on a horse jumps over a broken down airplane.	
Hypothesis: A person is at a diner, ordering an omelette.	
Label: contradiction

Premise: A person on a horse jumps over a broken down airplane.	
Hypothesis: A person is outdoors, on a horse.	
Label: entailment

Premise: Children smiling and waving at camera
Hypothesis: They are smiling at their parents
Label: neutral

Premise: Children smiling and waving at camera
Hypothesis: There are children present
Label: entailment

Premise: Children smiling and waving at camera
Hypothesis: The kids are frowning	
Label: contradiction

Premise: A boy is jumping on skateboard in the middle of a red bridge.	
Hypothesis: The boy skates down the sidewalk.
Label: contradiction

Premise: A boy is jumping on skateboard in the middle of a red bridge.
Hypothesis: The boy does a skateboarding trick.	
Label: entailment

Premise: A boy is jumping on skateboard in the middle of a red bridge.
Hypothesis: The boy is wearing safety equipment.
Label: neutral

{prompt}
"""

esnli_extra = """
"""

esnli_premise_hypothesis_rationale_label_template = """
Premise: A person on a horse jumps over a broken down airplane.	
Hypothesis: A person is training his horse for a competition.
Label: the person is not necessarily training his horse. So the answer is neutral.

Premise: A person on a horse jumps over a broken down airplane.	
Hypothesis: A person is at a diner, ordering an omelette.	
Label: One cannot be on a jumping horse cannot be a diner ordering food. So the answer is contradiction.

Premise: A person on a horse jumps over a broken down airplane.	
Hypothesis: A person is outdoors, on a horse.	
Label: a broken down airplane is outdoors. So the answer is entailment.

{prompt}
"""

def eval_acc_esnli(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)

    # List to store negative log likelihoods
    nlls = []
    if verbose:
        print(f"n_batch {n_batch}")

    num_correct = 0

    labels = ['entailment', 'neutral', 'contradiction']

    # Loop through each batch
    for i in range(n_batch): #range(nsamples)
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        # Calculate end index
        #j = min(i+bs, nsamples)
        
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if eval_rationale:
                prompt = esnli_premise_hypothesis_rationale_label_template.format(prompt=test_prompt[index_element]).strip()
            else:
                prompt = esnli_premise_hypothesis_label_template.format(prompt=test_prompt[index_element]).strip()
            
            batch.append(prompt)
            
            
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 


        # Forward pass through the model
        if eval_rationale:
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)#[0][input_ids.shape[1]:].unsqueeze(0)
        else:
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20) #20



        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if eval_rationale:
                output_decoded_first_line = output_i.split('\n\n')[0].strip()
                output_decoded_last_sentence = output_decoded_first_line.split(".")[-2].strip()
                output_decoded_clean = output_decoded_last_sentence.split('So the answer is ')[-1].strip()
            else:
                output_decoded_clean = output_i.split('\n\n')[0].strip()

            label = test_answer[i*batch_size + k]

            try:
                if output_decoded_clean == label:
                    num_correct += 1
            except:
                pass


    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

rte_premise_hypothesis_label_template = """
Premise: No Weapons of Mass Destruction Found in Iraq Yet.
Hypothesis: Weapons of Mass Destruction Found in Iraq.
Label: not entailment

Premise: A place of sorrow, after Pope John Paul II died, became a place of celebration, as Roman Catholic faithful gathered in downtown Chicago to mark the installation of new Pope Benedict XVI.	
Hypothesis: Pope Benedict XVI is the new leader of the Roman Catholic Church.
Label: entailment

Premise: Herceptin was already approved to treat the sickest breast cancer patients, and the company said, Monday, it will discuss with federal regulators the possibility of prescribing the drug for more breast cancer patients.	
Hypothesis: Herceptin can be used to treat breast cancer.
Label: entailment

{prompt}
"""

def eval_acc_rte(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)

    # List to store negative log likelihoods
    #if verbose:
    #    print(f"n_batch {n_batch}")

    num_correct = 0


    # Loop through each batch
    for i in range(n_batch): #range(nsamples)
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        # Calculate end index
        #j = min(i+bs, nsamples)

        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))


        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if shot == 'few':
                prompt = rte_premise_hypothesis_label_template.format(prompt=test_prompt[index_element]).strip()
            elif shot == 'zero':
                prompt = test_prompt[index_element]
            
            batch.append(prompt)
            
            
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 


        if shot == 'few':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20) #20
        elif shot == 'zero':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask)#[0][input_ids.shape[1]:].unsqueeze(0)


        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if shot == 'few':
                output_decoded_clean = output_i.split('\n\n')[0].strip()
            elif shot == 'zero':
                output_decoded_clean = output_i.strip()

            label = test_answer[i*batch_size + k]

            try:
                if output_decoded_clean == label:
                    num_correct += 1
            except:
                pass
        

    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

boolq_question_answer_template = """
Question: Persian (/ˈpɜːrʒən, -ʃən/), also known by its endonym Farsi (فارسی fārsi (fɒːɾˈsiː) ( listen)), is one of the Western Iranian languages within the Indo-Iranian branch of the Indo-European language family. It is primarily spoken in Iran, Afghanistan (officially known as Dari since 1958), and Tajikistan (officially known as Tajiki since the Soviet era), and some other regions which historically were Persianate societies and considered part of Greater Iran. It is written in the Persian alphabet, a modified variant of the Arabic script, which itself evolved from the Aramaic alphabet. do iran and afghanistan speak the same language	
Label: true

Question: Good Samaritan laws offer legal protection to people who give reasonable assistance to those who are, or who they believe to be, injured, ill, in peril, or otherwise incapacitated. The protection is intended to reduce bystanders' hesitation to assist, for fear of being sued or prosecuted for unintentional injury or wrongful death. An example of such a law in common-law areas of Canada: a good Samaritan doctrine is a legal principle that prevents a rescuer who has voluntarily helped a victim in distress from being successfully sued for wrongdoing. Its purpose is to keep people from being reluctant to help a stranger in need for fear of legal repercussions should they make some mistake in treatment. By contrast, a duty to rescue law requires people to offer assistance and holds those who fail to do so liable. do good samaritan laws protect those who help at an accident
Label: true

Question: As with other games in The Elder Scrolls series, the game is set on the continent of Tamriel. The events of the game occur a millennium before those of The Elder Scrolls V: Skyrim and around 800 years before The Elder Scrolls III: Morrowind and The Elder Scrolls IV: Oblivion. It has a broadly similar structure to Skyrim, with two separate conflicts progressing at the same time, one with the fate of the world in the balance, and one where the prize is supreme power on Tamriel. In The Elder Scrolls Online, the first struggle is against the Daedric Prince Molag Bal, who is attempting to meld the plane of Mundus with his realm of Coldharbour, and the second is to capture the vacant imperial throne, contested by three alliances of the mortal races. The player character has been sacrificed to Molag Bal, and Molag Bal has stolen their soul, the recovery of which is the primary game objective. is elder scrolls online the same as skyrim	
Label: false

{prompt}
"""

def eval_acc_boolq(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)

    # List to store negative log likelihoods
    #nlls = []
    #if verbose:
    #    print(f"nsamples {nsamples}")

    num_correct = 0

    # Loop through each batch
    for i in range(n_batch): #range(nsamples)
        
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        # Calculate end index
        #j = min(i+bs, nsamples)
        
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if shot == 'few':
                prompt = boolq_question_answer_template.format(prompt=test_prompt[index_element]).strip()
            elif shot == 'zero':
                prompt = test_prompt[index_element]
            
            batch.append(prompt)
            
            
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 

        # Forward pass through the model
        if shot == 'few':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20) #20 #[0][input_ids.shape[1]:].unsqueeze(0)
        elif shot == 'zero':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask)#[0][input_ids.shape[1]:].unsqueeze(0)


        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if shot == 'few':
                output_decoded_clean = output_i.split('\n')[0].strip()
            elif shot == 'zero':
                output_decoded_clean = output_i.strip()

            label = test_answer[i*batch_size + k]

            try:
                if output_decoded_clean.lower() == str(label).lower():
                    num_correct += 1
            except:
                pass


    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

commonsense_qa_question_answer_template = """
Question: Sammy wanted to go to where the people were. Where might he go?
Choices:
A. race track
B. populated areas
C. the desert
D. apartment
E. roadblock
Answer: B

Question: To locate a choker not located in a jewelry box or boutique where would you go? 
Choices: 
A. jewelry store
B. neck
C. jewlery box
D. jewelry box
E. boutique
Answer: A

Question: Google Maps and other highway and street GPS services have replaced what?
Choices:
A. united states
B. mexico
C. countryside
D. atlas
E. oceans
Answer: D

{prompt}
"""
def eval_acc_commonsense_qa(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)

    # List to store negative log likelihoods
    #nlls = []
    #if verbose:
    #    print(f"nsamples {nsamples}")

    num_correct = 0

    # Loop through each batch
    for i in range(n_batch): #range(nsamples)
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        # Calculate end index
        #j = min(i+bs, nsamples)
        
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if shot == 'few':
                prompt = commonsense_qa_question_answer_template.format(prompt=test_prompt[index_element]).strip()
            elif shot == 'zero':
                prompt = test_prompt[index_element]
            
            batch.append(prompt)
        


        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 
        

        # Forward pass through the model

        if shot == 'few':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20) #20 #[0][input_ids.shape[1]:].unsqueeze(0)
        elif shot == 'zero':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask)#[0][input_ids.shape[1]:].unsqueeze(0)


        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if shot == 'few':
                output_decoded_clean = output_i.split('\n')[0].strip()
            elif shot == 'zero':
                output_decoded_clean = output_i.strip()

            label = test_answer[i*batch_size + k]

            try:
                if output_decoded_clean.lower() == str(label).lower():
                    num_correct += 1
            except:
                pass


    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

race_question_answer_template = """
Article: Last week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects they thought they had. Given that I teach students who are training to be doctors, I was surprised do find that most thought that they would not be able to get the jobs they wanted without "outside help". "What kind of help is that?" I asked, expecting them to tell me that they would need a or family friend to help them out. "Surgery ," one replied. I was pretty alarmed by that response. It seems that the graduates of today are increasingly willing to go under the knife to get ahead of others when it comes to getting a job . One girl told me that she was considering surgery to increase her height. "They break your legs, put in special extending screws, and slowly expand the gap between the two ends of the bone as it re-grows, you can get at least 5 cm taller!" At that point, I was shocked. I am short, I can't deny that, but I don't think I would put myself through months of agony just to be a few centimetres taller. I don't even bother to wear shoes with thick soles, as I'm not trying to hide the fact that I am just not tall! It seems to me that there is a trend towards wanting "perfection" , and that is an ideal that just does not exist in reality. No one is born perfect, yet magazines, TV shows and movies present images of thin, tall, beautiful people as being the norm. Advertisements for slimming aids, beauty treatments and cosmetic surgery clinics fill the pages of newspapers, further creating an idea that "perfection" is a requirement, and that it must be purchased, no matter what the cost. In my opinion, skills, rather than appearance, should determine how successful a person is in his/her chosen career.
Question: We can know from the passage that the author works as a_.
Options:
A. doctor
B. model
C. teacher
D. reporter
Answer: C

Article: Last week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects they thought they had. Given that I teach students who are training to be doctors, I was surprised do find that most thought that they would not be able to get the jobs they wanted without "outside help". "What kind of help is that?" I asked, expecting them to tell me that they would need a or family friend to help them out. "Surgery ," one replied. I was pretty alarmed by that response. It seems that the graduates of today are increasingly willing to go under the knife to get ahead of others when it comes to getting a job . One girl told me that she was considering surgery to increase her height. "They break your legs, put in special extending screws, and slowly expand the gap between the two ends of the bone as it re-grows, you can get at least 5 cm taller!" At that point, I was shocked. I am short, I can't deny that, but I don't think I would put myself through months of agony just to be a few centimetres taller. I don't even bother to wear shoes with thick soles, as I'm not trying to hide the fact that I am just not tall! It seems to me that there is a trend towards wanting "perfection" , and that is an ideal that just does not exist in reality. No one is born perfect, yet magazines, TV shows and movies present images of thin, tall, beautiful people as being the norm. Advertisements for slimming aids, beauty treatments and cosmetic surgery clinics fill the pages of newspapers, further creating an idea that "perfection" is a requirement, and that it must be purchased, no matter what the cost. In my opinion, skills, rather than appearance, should determine how successful a person is in his/her chosen career.
Question: Many graduates today turn to cosmetic surgery to_.
Options: 
A. marry a better man/woman
B. become a model
C. get an advantage over others in job-hunting
D. attract more admirers
Answer: C

Article: Last week I talked with some of my students about what they wanted to do after they graduated, and what kind of job prospects they thought they had. Given that I teach students who are training to be doctors, I was surprised do find that most thought that they would not be able to get the jobs they wanted without "outside help". "What kind of help is that?" I asked, expecting them to tell me that they would need a or family friend to help them out. "Surgery ," one replied. I was pretty alarmed by that response. It seems that the graduates of today are increasingly willing to go under the knife to get ahead of others when it comes to getting a job . One girl told me that she was considering surgery to increase her height. "They break your legs, put in special extending screws, and slowly expand the gap between the two ends of the bone as it re-grows, you can get at least 5 cm taller!" At that point, I was shocked. I am short, I can't deny that, but I don't think I would put myself through months of agony just to be a few centimetres taller. I don't even bother to wear shoes with thick soles, as I'm not trying to hide the fact that I am just not tall! It seems to me that there is a trend towards wanting "perfection" , and that is an ideal that just does not exist in reality. No one is born perfect, yet magazines, TV shows and movies present images of thin, tall, beautiful people as being the norm. Advertisements for slimming aids, beauty treatments and cosmetic surgery clinics fill the pages of newspapers, further creating an idea that "perfection" is a requirement, and that it must be purchased, no matter what the cost. In my opinion, skills, rather than appearance, should determine how successful a person is in his/her chosen career.	
Question: 	
According to the passage, the author believes that_.
Options: 
A. everyone should purchase perfection, whatever the cost
B. it's right for graduates to ask for others to help them out in hunting for jobs
C. it is one's appearance instead of skills that really matters in one's career
D. media are to blame for misleading young people in their seeking for surgery
Answer: D

{prompt}
"""
def eval_acc_race(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)


    # List to store negative log likelihoods
    #nlls = []
    #if verbose:
    #    print(f"nsamples {nsamples}")

    num_correct = 0

    # Loop through each batch
    for i in range(n_batch): #range(nsamples)
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        # Calculate end index
        #j = min(i+bs, nsamples)
        
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))


        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if shot == 'few':
                prompt = race_question_answer_template.format(prompt=test_prompt[index_element]).strip()
            elif shot == 'zero':
                prompt = test_prompt[index_element]
            
            batch.append(prompt)
            
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 


        # Forward pass through the model
        if shot == 'few':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20) #20 [0][input_ids.shape[1]:].unsqueeze(0)
        elif shot == 'zero':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask)#[0][input_ids.shape[1]:].unsqueeze(0)


        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if shot == 'few':
                output_decoded_clean = output_i.split('\n')[0].strip()
            elif shot == 'zero':
                output_decoded_clean = output_i.strip()

            label = test_answer[i*batch_size + k]

            try:
                if output_decoded_clean.lower() == str(label).lower():
                    num_correct += 1
            except:
                pass


    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

winogrande_question_answer_template = """
Sentence: We can know from the passage that the author works as a_.
Options:
1. garage
2. backyard
Answer: 1

Sentence: The doctor diagnosed Justin with bipolar and Robert with anxiety. _ had terrible nerves recently.
Options: 
1. Justin
2. Robert
Answer: 2

Sentence: Dennis drew up a business proposal to present to Logan because _ wants his investment.
Options: 
1. Dennis
2. Logan
Answer: 1

{prompt}
"""
def eval_acc_winogrande(model, test_prompt, test_answer, tokenizer, sqlen, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    # Get input IDs
    #testenc = testenc.input_ids
    #print("testenc.size()", test_prompt.size())
    #verbose = True

    # Calculate number of samples
    n_batch = math.ceil(len(test_prompt)/batch_size)
    n_samples = len(test_prompt)


    # List to store negative log likelihoods
    #nlls = []
    #if verbose:
    #    print(f"nsamples {nsamples}")

    num_correct = 0

    # Loop through each batch
    for i in range(n_batch): #range(nsamples)
        #if verbose and i % 50 == 0:
        #    print(f"sample {i}")

        # Calculate end index
        #j = min(i+bs, nsamples)

        # Prepare inputs and move to device
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            if shot == 'few':
                prompt = winogrande_question_answer_template.format(prompt=test_prompt[index_element]).strip()
            elif shot == 'zero':
                prompt = test_prompt[index_element]
            
            batch.append(prompt)
        
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)
        attention_mask = tokenized_input.attention_mask.to(device) 

        if shot == 'few':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=20)#[0][input_ids.shape[1]:].unsqueeze(0)
        elif shot == 'zero':
            pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask)#[0][input_ids.shape[1]:].unsqueeze(0)


        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            if shot == 'few':
                output_decoded_clean = output_i.split('\n')[0].strip()
            elif shot == 'zero':
                output_decoded_clean = output_i.strip()

            label = test_answer[i*batch_size + k]

            try:
                if output_decoded_clean.lower() == str(label).lower():
                    num_correct += 1
            except:
                pass

    acc = num_correct / n_samples
    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return acc

# ---------------------------------- END ACC ---------------------------------- #




# ---------------------------------- BLEU ---------------------------------- #


BLEU_translation_template = """
English: Resumption of the session
French: Reprise de la session

English: I declare resumed the session of the European Parliament adjourned on Friday 17 December 1999, and I would like once again to wish you a happy new year in the hope that you enjoyed a pleasant festive period.
French: Je déclare reprise la session du Parlement européen qui avait été interrompue le vendredi 17 décembre dernier et je vous renouvelle tous mes vux en espérant que vous avez passé de bonnes vacances.

English: Although, as you will have seen, the dreaded 'millennium bug' failed to materialise, still the people in a number of countries suffered a series of natural disasters that truly were dreadful.
French: Comme vous avez pu le constater, le grand \"bogue de l'an 2000\" ne s'est pas produit. En revanche, les citoyens d'un certain nombre de nos pays ont été victimes de catastrophes naturelles qui ont vraiment été terribles.

{prompt}
"""



def eval_bleu(args, model, tokenizer, sqlen, device=torch.device("cuda:0"), 
        dataset="wmt14", shot='few', verbose=False, batch_size=1):
    print(f'evaluating on {dataset}')

    # Get the test loader
    _, test_data, pad_token = get_loaders(
        dataset, seed=0, seqlen=sqlen, tokenizer=tokenizer, train=False
    )
    if test_data:
        test_prompt, test_answer = test_data

    # Evaluate acc in no grad context to avoid updating the model
    start_time = time.time()
    
    with torch.no_grad():
        bleu = eval_bleu_dataset(model, test_prompt, test_answer, tokenizer, batch_size=batch_size[dataset], device=device, pad_token=pad_token, shot=shot, eval_rationale=args.eval_rationale, verbose=verbose)
    
    if verbose:
        print("execution time:", time.time() - start_time, flush=True)
    
    return bleu


def eval_bleu_dataset(model, test_prompt, test_answer, tokenizer, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    bleu = evaluate.load('bleu')
    nsamples = math.ceil(len(test_prompt)/batch_size)

    # List to store negative log likelihoods
    if verbose:
        print(f"nsamples {nsamples}", flush=True)

    predictions = []
    
    # Loop through each batch
    for i in range(nsamples): #nsamples
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            prompt = BLEU_translation_template.format(prompt=test_prompt[index_element]).strip()
            batch.append(prompt)
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device)#.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)#.unsqueeze(0)
        attention_mask = tokenized_input.attention_mask.to(device)#.unsqueeze(0) 
        
        pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)

        for k in range(this_batch_size):
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            output_decoded_clean = output_i.split('\n')[0].strip()
            predictions.append(output_decoded_clean)
    
    try:
        bleu_val = bleu.compute(predictions=predictions, references=test_answer)
    except Exception as e:
        print(f"Error computing BLEU: {e}")
        bleu_val = {'bleu': 0.0}

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()

    return bleu_val


# ---------------------------------- END BLEU ---------------------------------- #






# ---------------------------------- CodeBERTScore ---------------------------------- #

# Function to evaluate perplexity (ppl) on a specified model and tokenizer
def eval_code(args, model, tokenizer, sqlen, device=torch.device("cuda:0"), dataset="wikitext2", verbose=False, batch_size=1):
    # Set datasetx
    #model.to(device)
    # Print status
    print(f"evaluating on {dataset}")
    
    # Get the test loader
    _, testloader, _ = get_loaders(
        dataset, seed=0, seqlen=sqlen, tokenizer=tokenizer, train=False
    )
    if testloader:
        test_prompt, test_answer = testloader

    # Evaluate ppl in no grad context to avoid updating the model
    start_time = time.time()
    with torch.no_grad():
        if dataset == "opc":
            f1 = eval_code_opc(model, test_prompt, test_answer, tokenizer, batch_size=batch_size["opc"], device=device, verbose=verbose) #batch_size=8
        elif dataset == "ds1000":
            f1 = eval_code_ds1000(model, test_prompt, test_answer, tokenizer, batch_size=batch_size["ds1000"], device=device, verbose=verbose) #batch_size=4
        elif dataset == "mbpp":
            f1 = eval_code_mbpp(model, test_prompt, test_answer, tokenizer, batch_size=batch_size["mbpp"], device=device, verbose=verbose)  #batch_size=8

    if verbose:
        print("execution time:", time.time() - start_time, flush=True)
    
    f1_mean = torch.mean(f1).item()
    
    return f1_mean 


code_template_opc = """
Instruction: Write a Python function that takes a list of integers and returns the sum of the even numbers in the list.
Output:
def sum_even_numbers(numbers):
    return sum(num for num in numbers if num % 2 == 0)

Instruction: Create a function that takes an integer n and returns the nth Fibonacci number.
Output:
def fibonacci(n):
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b

Instruction: Write a function that takes a string and returns the string reversed.
Output:
def reverse_string(s):
    return s[::-1]

Instruction: {prompt}
""" 


def eval_code_opc(model, test_prompt, test_answer, tokenizer, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    nsamples = math.ceil(len(test_prompt)/batch_size)

    # List to store negative log likelihoods
    if verbose:
        print(f"nsamples {nsamples}", flush=True)

    predictions = []
    
    
    # Loop through each batch
    for i in range(nsamples): #nsamples
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            prompt = code_template_opc.format(prompt=test_prompt[index_element]).strip()
            batch.append(prompt)
        
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device) #.unsqueeze(0)
        attention_mask = tokenized_input.attention_mask.to(device) #.unsqueeze(0) 
        
        pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)

        for k in range(this_batch_size):
            #print(f"\n\n\ninput_i: {batch[k]}", flush=True)
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            #print(f"\noutput_i: {output_i}", flush=True)
            
            if 'Output:' in output_i:
                if 'Instruction:' in output_i:
                    output_decoded_clean = output_i.split('Output:')[1].split('Instruction:')[0].strip()
                else:
                    output_decoded_clean = output_i.split('Output:')[1].strip()
            else:
                output_decoded_clean = output_i.strip()
            
            #print(f"output_decoded_clean: {output_decoded_clean}", flush=True)
            predictions.append(output_decoded_clean)
        
    precision, recall, F1, F3 = code_bert_score.score(cands=predictions, refs=test_answer, lang='python')

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    
    return F1



code_template_ds1000 = """
Problem: 
I have a list of number like this:
[1, 2, 3, 4, 5]. 
I want to find the sum of all even numbers in this list. 
How to do this in Python?
The result should be 6.

A: <code> import numpy as np 
array = np.array([1, 2, 3, 4, 5])
sum_even = ... # put solution in this variable BEGIN SOLUTION <code>

Output:
sum_even = np.sum(array[array % 2 == 0])



Problem:
There is a list of strings like this:
['apple', 'banana', 'pear', 'date'].
I want to find the longest string in this list.
The result should be 'banana'.

A: <code> import numpy as np
array = np.array(['apple', 'banana', 'pear', 'date'])
longest_string = ... # put solution in this variable BEGIN SOLUTION <code>

Output:
longest_string = array[np.argmax([len(s) for s in array])]



Problem:
I have a dataframe with columns 'A', 'B', and 'C' like this:
A  B  C
1  2  3
4  5  6
7  8  9
I want to calculate the sum of each column.
The result should be {'A': 12, 'B': 15, 'C': 18}.

A: <code> import pandas as pd
data = {'A': [1, 4, 7], 'B': [2, 5, 8], 'C': [3, 6, 9]}
df = pd.DataFrame(data)
column_sums = ... # put solution in this variable BEGIN SOLUTION<code> 

Output:
column_sums = df.sum().to_dict()



{prompt}
""" 




code_template_ds1000 = """
Problem: 
I have a list of number like this:
[1, 2, 3, 4, 5]. 
I want to find the sum of all even numbers in this list. 
How to do this in Python?
The result should be 6.

A: <code> import numpy as np 
array = np.array([1, 2, 3, 4, 5])
</code>
sum_even = ... # put solution in this variable 
BEGIN SOLUTION 
<code>
sum_even = np.sum(array[array % 2 == 0])
</code>


Problem:
There is a list of strings like this:
['apple', 'banana', 'pear', 'date'].
I want to find the longest string in this list.
The result should be 'banana'.

A: <code> import numpy as np
array = np.array(['apple', 'banana', 'pear', 'date'])
<code>
longest_string = ... # put solution in this variable 
BEGIN SOLUTION 
<code>
longest_string = array[np.argmax([len(s) for s in array])]
</code>


Problem:
I have a matrix like this:
1  2  3
4  5  6
7  8  9
I want to calculate the sum of each row.
The result should be []

A: <code> import numpy as np
data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
data = np.array(data)
<code> 
column_sums = ... # put solution in this variable 
BEGIN SOLUTION
<code>
column_sums = data.sum(axis=1).tolist()
</code>


{prompt}
""" 

def eval_code_ds1000(model, test_prompt, test_answer, tokenizer, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    nsamples = math.ceil(len(test_prompt)/batch_size)

    # List to store negative log likelihoods
    if verbose:
        print(f"nsamples {nsamples}", flush=True)

    predictions = []
    
    
    # Loop through each batch
    for i in range(nsamples): #nsamples
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            prompt = code_template_ds1000.format(prompt=test_prompt[index_element]).strip()
            batch.append(prompt)
        
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device) #.unsqueeze(0)
        attention_mask = tokenized_input.attention_mask.to(device) #.unsqueeze(0) 
        
        pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)

        for k in range(this_batch_size):
            #print(f"\n\n\ninput_i: {batch[k]}", flush=True)
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            #print(f"\noutput_i: {output_i}", flush=True)
            
            if '</code>' in output_i:
                output_decoded_clean = output_i.split('</code>')[0].strip()
            else:
                output_decoded_clean = output_i.strip()
            
            #print(f"output_decoded_clean: {output_decoded_clean}", flush=True)
            predictions.append(output_decoded_clean)
    
    precision, recall, F1, F3 = code_bert_score.score(cands=predictions, refs=test_answer, lang='python')

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    
    return F1




def eval_code_mbpp(model, test_prompt, test_answer, tokenizer, pad_token=None, batch_size=1, device=None, shot='few', eval_rationale=False, verbose=False):
    nsamples = math.ceil(len(test_prompt)/batch_size)

    # List to store negative log likelihoods
    if verbose:
        print(f"nsamples {nsamples}", flush=True)

    predictions = []
    
    
    # Loop through each batch
    for i in range(nsamples): #nsamples
        this_batch_size = min(batch_size, abs(len(test_prompt) - i*batch_size))
        
        batch = []
        for k in range(this_batch_size):
            index_element = i*batch_size + k
            
            prompt = code_template_opc.format(prompt=test_prompt[index_element]).strip()
            batch.append(prompt)
        
        
        tokenized_input = tokenizer(batch, return_tensors='pt', padding='longest').to(device) #.input_ids.to(device)
        input_ids = tokenized_input.input_ids.to(device)#.unsqueeze(0)
        attention_mask = tokenized_input.attention_mask.to(device)#.unsqueeze(0) 
        
        pred_ids_batch = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=256)

        for k in range(this_batch_size):
            #print(f"\n\n\ninput_i: {batch[k]}", flush=True)
            pred_ids_i = pred_ids_batch[k][input_ids.shape[1]:].unsqueeze(0)
            output_i = tokenizer.batch_decode(pred_ids_i, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            #print(f"\noutput_i: {output_i}", flush=True)
            
            if 'Output:' in output_i:
                if 'Instruction:' in output_i:
                    output_decoded_clean = output_i.split('Output:')[1].split('Instruction:')[0].strip()
                else:
                    output_decoded_clean = output_i.split('Output:')[1].strip()
            else:
                output_decoded_clean = output_i.strip()
            
            #print(f"output_decoded_clean: {output_decoded_clean}", flush=True)
            predictions.append(output_decoded_clean)
    
    precision, recall, F1, F3 = code_bert_score.score(cands=predictions, refs=test_answer, lang='python')

    # Empty CUDA cache to save memory
    torch.cuda.empty_cache()
    
    return F1
