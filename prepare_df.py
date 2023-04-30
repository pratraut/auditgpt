import pandas as pd
import fasttext
import pickle
import os

from transformers import AutoTokenizer, AutoModelForMaskedLM
from bert_score import BERTScorer
from tqdm import tqdm
from audit_utils import *

def embed_code(code):
    inputs = tokenizer(code, padding=True, truncation=True, return_tensors='pt')
    #tokens = scorer.get_embedding(code)
    outputs = model(**inputs, output_hidden_states=True)
    embeddings = outputs.hidden_states[-1].mean(dim=1).detach().numpy()
    return embeddings


def embed_code_with_progress(p, desc):
    embeddings = []
    for code in tqdm(p, desc=desc, unit='snippet'):
        embedding = embed_code(code)
        embeddings.append(embedding)
    return embeddings

tokenizer = AutoTokenizer.from_pretrained("neulab/codebert-javascript")
model = AutoModelForMaskedLM.from_pretrained("neulab/codebert-javascript")
scorer = BERTScorer(model_type="neulab/codebert-javascript")

issues_csv = ['issues_repos_data_sherlock-audit_issues', 'issues_repos_data_code-423n4_issues']

pairs_concatenated = pd.DataFrame()
for issue_csv in issues_csv:
    
    pairs_csv = pickle.load(open(f'data/{issue_csv}.pickle','rb'))

    pairs = pd.read_csv(issue_csv+".csv", delimiter='\t', names=['name','code','expl'])
    pairs.dropna(inplace=True)
    pairs['code'] = pairs['code'].replace('\s+', ' ', regex=True)
    pairs.drop_duplicates(['code','expl'], inplace=True)

    pairs_concatenated = pd.concat([pairs_concatenated, pairs], axis=0)

pairs = pairs_concatenated
pairs.drop_duplicates(['code','expl'], inplace=True)

pairs['reserved_words'] = pairs['code'].apply(lambda x: extract_words(x, reserved_words))
pairs['variable_types'] = pairs['code'].apply(lambda x: extract_words(x, variable_types))
pairs['common_keywords'] = pairs['code'].apply(lambda x: extract_words(x, common_keywords))
pairs['strings'] = pairs['code'].apply(lambda x: extract_strings(x))

pairs['code_sanitized'] = pairs['code'].apply(lambda x: sanitize_code(x, reserved_words, variable_types, common_keywords))

try:
    os.remove(f'data/code_ft_full.txt')
except:
    pass
with open(f'data/code_ft_full.txt', 'w+') as f:
    f.write('\n\n'.join(pairs['code'].astype(str).tolist()))
try:
    os.remove(f'data/code_sanitized_ft_full.txt')
except:
    pass
with open(f'data/code_sanitized_ft_full.txt', 'w+') as f:
    f.write('\n\n'.join(pairs['code_sanitized'].astype(str).tolist()))

ft_model = fasttext.train_unsupervised(f'data/code_ft_full.txt', dim=100)
try:
    os.remove(f'data/fasttext_model_full.bin')
except:
    pass
ft_model.save_model(f'data/fasttext_model_full.bin')
pairs['embedding_ft'] = pairs['code'].astype(str).apply(lambda x: ft_model.get_sentence_vector(x))

ft_model = fasttext.train_unsupervised(f'data/code_sanitized_ft_full.txt', dim=100)
try:
    os.remove(f'data/fasttext_model_sanitized_full.bin')
except:
    pass
ft_model.save_model(f'data/fasttext_model_sanitized_full.bin')
pairs['embedding_sanitized_ft'] = pairs['code_sanitized'].astype(str).apply(lambda x: ft_model.get_sentence_vector(x))

#pairs['embedding_codebert'] = pairs['code'].apply(embed_code)
pairs['embedding_codebert'] = embed_code_with_progress(pairs['code'],'embedding_codebert')
#pairs['embedding_sanitized_codebert'] = pairs['code_sanitized'].apply(embed_code)
pairs['embedding_sanitized_codebert'] = embed_code_with_progress(pairs['code_sanitized'],'embedding_sanitized_codebert')

with open(f'data/pairs_full.pickle', 'wb+') as f:
    pickle.dump(pairs, f)