import json
import logging, os
logging.disable(logging.WARNING)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import pickle
import random
import sys
import string
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
import warnings

import fasttext
fasttext.FastText.eprint = lambda x: None

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from transformers import AutoTokenizer, AutoModelForMaskedLM

from audit_utils import *
from bert_score import BERTScorer
from audit_utils import _pack_batch
import gmn
from gmn.evaluation import compute_similarity
from gmn.utils import load_pickles_from_directory, reshape_and_split_tensor
from gmn.configure import get_default_config

from sklearn.metrics.pairwise import cosine_similarity

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
sys.setrecursionlimit(10000) 

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

nltk.download('stopwords', download_dir = os.getenv("DATA_DIR"), quiet=True)
nltk.download('punkt', download_dir = os.getenv("DATA_DIR"), quiet=True)

import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel

from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # or "true" depending on your needs


# Filter warnings from the fasttext library
warnings.filterwarnings("ignore", category=UserWarning, module="fasttext")

# tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
# model = RobertaModel.from_pretrained("microsoft/codebert-base")

# tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT")
# model = AutoModelForMaskedLM.from_pretrained("microsoft/CodeBERT")
# scorer = BERTScorer(model_type="microsoft/CodeBERT")

tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-javascript")
model = AutoModelForMaskedLM.from_pretrained("neulab/codebert-javascript")
scorer = BERTScorer(model_type="neulab/codebert-javascript")

ft_model_code = fasttext.load_model('data/fasttext_model_full.bin')
ft_model_code_sanitized = fasttext.load_model('data/fasttext_model_sanitized_full.bin')

pairs = pickle.load(open('data/pairs_full.pickle','rb'))

# create a NearestNeighbors object
nn_ft = NearestNeighbors(n_neighbors=len(pairs['embedding_ft'])-1, algorithm='brute', metric='cosine') #algorithm='ball_tree') #n_neighbors=5
nn_bert = NearestNeighbors(n_neighbors=len(pairs['embedding_codebert'])-1, algorithm='brute', metric='cosine') #algorithm='ball_tree') #n_neighbors=5
nn_sanitized_ft = NearestNeighbors(n_neighbors=len(pairs['embedding_sanitized_ft'])-1, algorithm='brute', metric='cosine') #algorithm='ball_tree') #n_neighbors=5
nn_sanitized_bert = NearestNeighbors(n_neighbors=len(pairs['embedding_sanitized_codebert'])-1, algorithm='brute', metric='cosine') #algorithm='ball_tree') #n_neighbors=5

# fit the NearestNeighbors object to the embeddings
np_embeddings_codebert = pairs['embedding_codebert'].apply(lambda x: np.array(x[0]))
np_embeddings_codebert = np.array([list(x) for x in np_embeddings_codebert])
X = np.stack(np_embeddings_codebert)
nn_bert.fit(X)

np_embeddings_fasttext = np.array([list(x) for x in pairs['embedding_ft']])
# fit the NearestNeighbors object to the embeddings
X = np.stack(np_embeddings_fasttext)
nn_ft.fit(X)

# fit the NearestNeighbors object to the embeddings
np_embeddings_sanitized_codebert = pairs['embedding_sanitized_codebert'].apply(lambda x: np.array(x[0]))
np_embeddings_sanitized_codebert = np.array([list(x) for x in np_embeddings_codebert])
X = np.stack(np_embeddings_sanitized_codebert)
nn_sanitized_bert.fit(X)

np_embeddings_sanitized_fasttext = np.array([list(x) for x in pairs['embedding_sanitized_ft']])
# fit the NearestNeighbors object to the embeddings
X = np.stack(np_embeddings_sanitized_fasttext)
nn_sanitized_ft.fit(X)

def generate_report(sol_code):

    retObj={}
    retObj['functions']={}

    # Print configure
    config = get_default_config()

    #config = copy.deepcopy(config)
    functions = separate_functions(sol_code)

    top_k = min(5, len(pairs['embedding_codebert']))
    top_n = 10

    for name, function_body in tqdm(functions.items(), desc='Processing contract functions', unit='function'):
    #for name, function_body in functions.items():

        function_body = function_body.replace("\n",' ')
        sanitized_function_body = sanitize_code(function_body)

        inputs = tokenizer(function_body, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs, output_hidden_states=True)
        embedding_bert = outputs.hidden_states[-1].mean(dim=1).detach().numpy()

        inputs = tokenizer(sanitized_function_body, padding=True, truncation=True, return_tensors='pt')
        outputs = model(**inputs, output_hidden_states=True)
        embedding_sanitized_bert = outputs.hidden_states[-1].mean(dim=1).detach().numpy()


        cos_scores_bert = cosine_similarity(embedding_bert, np_embeddings_codebert)
        cos_scores_sanitized_bert = cosine_similarity(embedding_sanitized_bert, np_embeddings_sanitized_codebert)
        
        
        ft_vector = ft_model_code.get_sentence_vector(function_body)
        cos_scores_ft = cosine_similarity(ft_vector.reshape(1,-1), np_embeddings_fasttext)

        ft_vector = ft_model_code_sanitized.get_sentence_vector(sanitized_function_body)
        cos_scores_sanitized_ft = cosine_similarity(ft_vector.reshape(1,-1), np_embeddings_sanitized_fasttext)

        # Combine the scores and maintain the original indices
        indexed_bert = list(enumerate(cos_scores_bert[0]))
        indexed_ft = list(enumerate(cos_scores_ft[0]))

        indexed_sanitized_bert = list(enumerate(cos_scores_sanitized_bert[0]))
        indexed_sanitized_ft = list(enumerate(cos_scores_sanitized_ft[0]))

        # Sort by scores in descending order
        sorted_scores_bert = sorted(indexed_bert, key=lambda x: x[1], reverse=True)
        sorted_scores_ft = sorted(indexed_ft, key=lambda x: x[1], reverse=True)

        sorted_scores_sanitized_bert = sorted(indexed_sanitized_bert, key=lambda x: x[1], reverse=True)
        sorted_scores_sanitized_ft = sorted(indexed_sanitized_ft, key=lambda x: x[1], reverse=True)


        sorted_pairs = calc_similarity_score(function_body, pairs)

        gpt_prompt = f"""
                    Please explain the following code snippet, focusing on the function's purpose and how it interacts with the storage, memory and other internal and external functions. Include explanations for each data type. Do not mention custom variable/function names. :

                    '{function_body}' """

        gpt_semantic_expl = send_prompt_to_chatgpt(gpt_prompt)

        gpt_prompt = f"Extract all technical terms and keywords from the following text. Write them in a comma separated list. Use infinitive form, when possible, one unique word at a time : \n\n \" {gpt_semantic_expl['choices'][0]['message']['content']} \" "

        gpt_word_list = send_prompt_to_chatgpt(gpt_prompt)

        single_words = []
        word_list = gpt_word_list['choices'][0]['message']['content'].split(",")
        word_list = [w.strip() for w in word_list if len(w.strip())>1]
        for w in word_list:
            if ' ' in w:
                single_words+=w.split(" ")
            else:
                single_words+=[w]
        single_words = list(set(single_words))
        relevant_words = []
        for w in single_words:
            if w in reserved_words or w in common_keywords or w in variable_types:
                relevant_words.append(w)
        
        relevant_words = list(set(relevant_words))

        kw_similarity_scores = kw_similarity_score(pairs, 'expl', relevant_words)
        list_expl = list(enumerate(kw_similarity_scores))
        list_expl = sorted(list_expl, key=lambda x: x[1], reverse=True)

        for sse in list_expl[:5]:
            issue = {}
            issue['title']=name
            issue['code']=pairs.iloc[sse[0]]['code']
            issue['explanation']=pairs.iloc[sse[0]]['expl']
            if name not in retObj['functions'].keys():
                retObj['functions'][name]=[]
                retObj['functions'][name].append(issue)
            else:
                retObj['functions'][name].append(issue)

        for ssb in sorted_scores_bert[:3]:
            issue = {}
            issue['title']=name
            issue['code']=pairs.iloc[ssb[0]]['code']
            issue['explanation']=pairs.iloc[ssb[0]]['expl']
            if name not in retObj['functions'].keys():
                retObj['functions'][name]=[]
                retObj['functions'][name].append(issue)
            else:
                retObj['functions'][name].append(issue)
        
        for ssft in sorted_scores_ft[:3]:
            issue = {}
            issue['title']=name
            issue['code']=pairs.iloc[ssft[0]]['code']
            issue['explanation']=pairs.iloc[ssft[0]]['expl']
            if name not in retObj['functions'].keys():
                retObj['functions'][name]=[]
                retObj['functions'][name].append(issue)
            else:
                retObj['functions'][name].append(issue)
        
        for ssb in sorted_scores_sanitized_bert[:3]:
            issue = {}
            issue['title']=name
            issue['code']=pairs.iloc[ssb[0]]['code_sanitized']
            issue['explanation']=pairs.iloc[ssb[0]]['expl']
            if name not in retObj['functions'].keys():
                retObj['functions'][name]=[]
                retObj['functions'][name].append(issue)
            else:
                retObj['functions'][name].append(issue)
        
        for ssft in sorted_scores_sanitized_ft[:3]:
            issue = {}
            issue['title']=name
            issue['code']=pairs.iloc[ssft[0]]['code_sanitized']
            issue['explanation']=pairs.iloc[ssft[0]]['expl']
            if name not in retObj['functions'].keys():
                retObj['functions'][name]=[]
                retObj['functions'][name].append(issue)
            else:
                retObj['functions'][name].append(issue)
        
        for sp in sorted_pairs[:3].values:
            issue = {}
            issue['title']=name
            issue['code']=sp[1]
            issue['explanation']=sp[2]
            if name not in retObj['functions'].keys():
                retObj['functions'][name]=[]
                retObj['functions'][name].append(issue)
            else:
                retObj['functions'][name].append(issue)

    return retObj

def remove_hardhat_imports(solidity_code: str) -> str:
    hardhat_import_pattern = r'^import\s+"hardhat/.*?";\s*$'
    return re.sub(hardhat_import_pattern, "", solidity_code, flags=re.MULTILINE)

def collect_sol_files(path: str, sol_files: list) -> None:
    for entry in os.listdir(path):
        entry_path = os.path.join(path, entry)

        if os.path.isfile(entry_path) and entry.endswith('.sol'):
            with open(entry_path, 'r') as f:
                original_code = f.read()

            cleaned_code = remove_hardhat_imports(original_code)

            with open(entry_path, 'w') as f:
                f.write(cleaned_code)

            sol_files.append(entry_path)
        elif os.path.isdir(entry_path):
            collect_sol_files(entry_path, sol_files)

def remove_stopwords(text: str) -> str:
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word.lower() not in stop_words]
    return ' '.join(filtered_text)

def sanitize_explanation(text):

    ## Lines of code
    ## Vulnerability details
    ## Proof of Concept
    ## Impact
    ## Tools Used
    ## Recommended Mitigation Steps
    # # Code Snippet
    # # Tool used
    # Manual Review
    # # Recommendation
    # # Summary
    # Handle

    pattern = r'\s*#{1,2}\s?#{0,2}\s*(Handle|Summary|Lines of code|Vulnerability details|Vulnerability detail|Proof of Concept|Impact|Tools Used|Recommended Mitigation Steps|Code Snippet|Tool used|Manual Review|Recommendation)\s*'
    text = re.sub(pattern, ' . ', text, flags=re.IGNORECASE)
    text = re.sub(r'medium |high |low | Manual Review', ' ', text, flags=re.IGNORECASE)


    #Remove newline
    text = text.replace("\n",' ')

    #Remove linefeed
    text = text.replace("\r",' ')

    # Remove [n]
    text = re.sub(r'\[n\]', ' ', text)

    # Remove [LINK]
    text = re.sub(r'\[LINK\]', ' ', text)

    # Remove code between ```
    text = re.sub(r'```.*?```', ' ', text, flags=re.DOTALL)

    # Remove code between `
    text = re.sub(r'`.*?`', ' ', text, flags=re.DOTALL)

    # Remove all [] with mentions of .sol files
    text = re.sub(r'\[[^]]*\.sol[^]]*\]', ' ', text)

    # Remove all mentions of .sol files
    text = re.sub(r'[^\s]+\.sol', ' ', text)

    #Remove commas, full-stops
    text = re.sub(r',|\.', '', text)

    # Remove multiple spaces
    text = re.sub(r'\s+',' ',text)

    return text

def compress_explanation(text):

    # Remove vowels (still works)
    text = re.sub(r'a|e|o|i|y|u|A|E|O|I|Y|U','',text)

    return text

def main():
    directory = f'{os.getenv("HOME_DIR")}/test'
    sol_files = []
    

    collect_sol_files(directory, sol_files)

    base_path = os.path.commonpath(sol_files)
    print('Processing Solidity (.sol) files:')
    for sol_file in sol_files:
        print('/'.join(sol_file.split("/")[-2:]))
        with open(sol_file, 'r', encoding='utf-8') as file:
            content = file.read()
            contract_obj = generate_report(content)

        detected_explanations = {}
        target_filename = directory+f'/report_{sol_file.split("/")[-1]}_functions_debug.md'
        for f_name in contract_obj['functions']:
            function_list = contract_obj['functions'][f_name]
            if len(function_list)>0:
                with open(target_filename, 'a+', encoding='utf-8') as file:
                    file.write(f"\n[function {function_list[0]['title']}]:\n\n")
                    if function_list[0]['title'] not in detected_explanations:
                        detected_explanations[function_list[0]['title']]=[]
                    for f_obj in function_list:
                        expl = sanitize_explanation(f_obj['explanation'])
                        expl = remove_stopwords(expl)
                        expl = compress_explanation(expl)
                        if expl not in detected_explanations[function_list[0]['title']]:
                            file.write(f">>>>> {expl}\n")
                            detected_explanations[function_list[0]['title']].append(expl)

    functions = separate_functions(content)
    target_filename = directory+f'/report_{sol_file.split("/")[-1]}_functions_gpt.md'
    for function, function_body in tqdm(functions.items(), desc='Feeding hints to GPT', unit='function'):
    #for function, function_body in functions.items():
        prompt_hints = "\n\n>>>".join(detected_explanations[function])
        prompt = f"""There's a potential issue with the code:\n\n{function_body}\n\nFind the three most relevant hints and rewrite it better with your own words. \n\nHints:\n\n{prompt_hints}"""
        api_key = os.getenv("OPENAI_KEY")
        try:
            response = send_prompt_to_chatgpt(prompt, api_key)
        except Exception as err:
            print(err)
            continue
        with open(target_filename, 'a+', encoding='utf-8') as file:
            file.write(f"\n\n[function {function}]:\n")
            file.write(f"{response['choices'][0]['message']['content']}\n")

if __name__ == '__main__':
    try:
        main()
    except Exception as err:
        print(err)