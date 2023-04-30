import re
import numpy as np
import PyPDF2
import os
import json
import os
import requests
import networkx as nx
import matplotlib.pyplot as plt
import collections
import os
import subprocess
import re

import pickle
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import openai

from dotenv import load_dotenv

load_dotenv()

def send_prompt_to_chatgpt(prompt, api_key=os.getenv('OPENAI_KEY')):
    openai.api_key = api_key
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # Replace with the appropriate engine for GPT-4 when available
            messages=[{"role": "user", "content": prompt}]
        )
    except openai.error.RateLimitError as rle:
        print(rle)
        raise rle
    
    return response

# Reserved words
reserved_words = [
        'abstract', 'after', 'address', 'alias', 'allow', 'anonymous', 'apply', 'as', 'assert', 'auto', 'before', 'begin',
        'case', 'catch', 'chain', 'check', 'class', 'const', 'constructor', 'continue', 'debug', 'default', 'delete', 'do',
        'else', 'elseif', 'emit', 'end', 'enum', 'event', 'external', 'fallback', 'false', 'final', 'fixed', 'for', 'from',
        'function', 'get', 'hash', 'if', 'implements', 'import', 'indexed', 'in', 'inline', 'interface', 'internal', 'is',
        'library', 'let', 'link', 'mapping', 'match', 'memory', 'modifier', 'new', 'not', 'null', 'of', 'off', 'on',
        'operator', 'override', 'package', 'payable', 'pragma', 'private', 'public', 'pure', 'read', 'receive', 'revert',
        'returns', 'selfdestruct', 'set', 'signed', 'sizeof', 'static', 'storage', 'string', 'struct', 'sub', 'super',
        'switch', 'symbol', 'then', 'this', 'throw', 'to', 'transaction', 'true', 'try', 'type', 'typedef', 'typeof', 'uint',
        'ulid', 'unalterable', 'unchecked', 'uniform', 'unsafe', 'unverified', 'using', 'view', 'virtual', 'when', 'while',
        'with'
    ]

# Variable types
variable_types = [
        'address', 'bool', 'string', 'int', 'uint', 'fixed', 'ufixed', 'bytes', 'bytesN', 'mapping', 'array', 'struct',
        'enum', 'int8', 'int16', 'int24', 'int32', 'int40', 'int48', 'int56', 'int64', 'int72', 'int80', 'int88', 'int96',
        'int104', 'int112', 'int120', 'int128', 'int136', 'int144', 'int152', 'int160', 'int168', 'int176', 'int184',
        'int192', 'int200', 'int208', 'int216', 'int224', 'int232', 'int240', 'int248', 'int256', 'uint8', 'uint16',
        'uint24', 'uint32', 'uint40', 'uint48', 'uint56', 'uint64', 'uint72', 'uint80', 'uint88', 'uint96', 'uint104',
        'uint112', 'uint120', 'uint128', 'uint136', 'uint144', 'uint152', 'uint160', 'uint168', 'uint176', 'uint184',
        'uint192', 'uint200', 'uint208', 'uint216', 'uint224', 'uint232', 'uint240', 'uint248', 'uint256'
    ]

# Commonly used keywords
common_keywords = [
        'payable', 'view', 'pure', 'external', 'internal', 'public', 'private', 'constant', 'memory', 'storage',
        'keccak256', 'sha256', 'revert', 'require', 'assert', 'selfdestruct', 'emit', 'fallback', 'constructor',
        'modifier', 'event', 'return', 'using', 'library', 'interface', 'import', 'from', 'for', 'while', 'if', 'else',
        'do', 'continue', 'break', 'try', 'catch', 'finally', 'throw', 'bytes', 'block', 'amount', 'payment', 'token',
        'contract', 'router', 'transfer', 'executionDelegate'
    ]


# Helper function to extract words from a string
def extract_words(text, words_list):
    return [word for word in words_list if re.search(r'\b' + word + r'\b', text)]

# Helper function to extract strings
def extract_strings(text):
    return re.findall(r'"[^"\\]*(?:\\.[^"\\]*)*"', text)

def jaccard_similarity(set1, set2):
    if len(set1) == 0 and len(set2) == 0:
        return 1.0
    intersection = set(set1).intersection(set(set2))
    union = set(set1).union(set(set2))
    return len(intersection) / len(union)

def count_difference_similarity(count1, count2):
    if count1 == 0 and count2 == 0:
        return 1.0
    return 1 - abs(count1 - count2) / max(count1, count2)

def calc_similarity_score(function_body, pairs):

    # Preprocess new_code to extract reserved words, variable types, common keywords, and strings
    function_reserved_words = extract_words(function_body, reserved_words)
    function_variable_types = extract_words(function_body, variable_types)
    function_common_keywords = extract_words(function_body, common_keywords)
    function_strings = extract_strings(function_body)

    def similarity_score(row):
        reserved_words_similarity = jaccard_similarity(row['reserved_words'], function_reserved_words)
        variable_types_similarity = jaccard_similarity(row['variable_types'], function_variable_types)
        common_keywords_similarity = jaccard_similarity(row['common_keywords'], function_common_keywords)
        strings_similarity = jaccard_similarity(row['strings'], function_strings)

        reserved_words_count_similarity = count_difference_similarity(len(row['reserved_words']), len(function_reserved_words))
        variable_types_count_similarity = count_difference_similarity(len(row['variable_types']), len(function_variable_types))
        common_keywords_count_similarity = count_difference_similarity(len(row['common_keywords']), len(function_common_keywords))
        strings_count_similarity = count_difference_similarity(len(row['strings']), len(function_strings))

        # Average the similarity scores
        avg_similarity = (reserved_words_similarity + variable_types_similarity + common_keywords_similarity + strings_similarity) / 4
        # Average the count similarity scores
        avg_count_similarity = (reserved_words_count_similarity + variable_types_count_similarity + common_keywords_count_similarity + strings_count_similarity) / 4

        # Combine both similarities (you can also use other methods like weighted average)
        return (avg_similarity*0.8 + avg_count_similarity*0.2) / 2

    # Calculate similarity scores for each row
    pairs['similarity_score'] = pairs.apply(similarity_score, axis=1)
    # Sort the DataFrame by similarity score in descending order
    sorted_pairs = pairs.sort_values(by='similarity_score', ascending=False)

    return sorted_pairs

def sanitize_code(code, reserved_words = reserved_words, variable_types = variable_types, common_keywords = common_keywords):
    # Combine all the keywords into a single set
    all_keywords = set(reserved_words + variable_types + common_keywords)

    # Function to replace non-keyword ASCII words with a single '#'
    def replace_non_keyword(match):
        word = match.group(0)
        return word if word in all_keywords else '#'

    # Replace each ASCII word not in the combined set with '#'
    sanitized_code = re.sub(r'\b[a-zA-Z_]+\b', replace_non_keyword, code)

    return sanitized_code

def search_embeddings(query_embedding, model, df, n_results=5):
    distances, indices = model.kneighbors([query_embedding], n_neighbors=n_results)
    return df.iloc[indices[0]]

def kw_similarity_score(dataframe, column_name, reference_list):
    # Convert the reference list to a single string
    reference_string = ', '.join(reference_list)

    # Create a new DataFrame containing the reference string and the input column data
    combined_data = pd.concat([pd.Series([reference_string]), dataframe[column_name]], ignore_index=True)

    # Calculate the term frequency-inverse document frequency (TF-IDF) matrix
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(combined_data)

    # Calculate the cosine similarity scores between the reference string and each text in the input column
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:])

    return similarity_scores.flatten()

def get_github_issue_content(issue_url, auth_token):
    # extract the repository owner, name, and issue number from the issue URL
    parts = issue_url.split('/')
    owner, repo, _, issue_number = parts[-4:]
    # construct the API URL for the issue
    api_url = f'https://api.github.com/repos/{owner}/{repo}/issues/{issue_number}'
    # make a GET request to the API with authentication
    headers = {'Authorization': f'token {auth_token}'}
    response = requests.get(api_url, headers=headers)
    # extract the body of the first post from the response
    data = json.loads(response.text)
    post_body = data['body']
    return post_body

def extract_text_from_pdfs(dir_path):
    """
    Extracts text from all PDF files in the specified directory and saves it in a .md format.
    :param dir_path: The path to the directory containing the PDF files.
    """
    # Get a list of all PDF files in the directory
    pdf_files = [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
    
    # Iterate over each PDF file and extract its text
    for pdf_file in pdf_files:
        with open(os.path.join(dir_path, pdf_file), 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Write the extracted text to a .md file with the same name as the PDF file
            with open(os.path.join(dir_path, pdf_file[:-4] + '.md'), 'w') as f:
                f.write(text)

def read_solidity_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def separate_functions(sol_code):
    functions_dict = {}

    # Remove comments
    sol_code = re.sub(re.compile('/\*.*?\*/', re.DOTALL), '', sol_code) # Remove block comments
    sol_code = re.sub(re.compile('//.*?\n'), '', sol_code) # Remove single-line comments

    # Find all function signatures
    function_signatures = re.findall(r'function\s+([a-zA-Z0-9_]+)\s*\(', sol_code)

    # Separate functions
    for function_name in function_signatures:
        function_pattern = re.compile(
            r'(function\s+' + function_name + r'\s*\(.*?\).*?\{.*?\})',
            re.DOTALL
        )
        function_body = re.search(function_pattern, sol_code)
        if function_body:
            functions_dict[function_name] = function_body.group(1)
    
    return functions_dict

def search_embeddings(query_embedding, model, df, n_results=5):
    distances, indices = model.kneighbors([query_embedding], n_neighbors=n_results)
    return df.iloc[indices[0]]

def traverse_and_convert_to_graph(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.sol_json.ast'):
                
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                os.makedirs(output_path, exist_ok=True)

                pickle_output_file = os.path.join(output_path, file.replace('.sol_json.ast', '.pickle'))

                #if 'arithmetic' not in pickle_output_file and 'access' not in pickle_output_file:
                #    continue

                if os.path.exists(pickle_output_file):
                    print(f'Skipped {file_path} as the pickle file already exists')
                    continue

                with open(file_path, 'r') as f:
                    solidity_ast_json = f.read()

                try:
                    asg = ASG(solidity_ast_json)
                    asg.generate()
                    graph = asg.to_networkx_graph()
                    
                    with open(pickle_output_file, 'wb') as pf:
                        pickle.dump(graph, pf)

                    print(f'Successfully converted and saved {file_path} as a pickle in {pickle_output_file}')
                except Exception as e:
                    print(f'Error converting {file_path} to a networkx Graph: {e}')

def run_solc_ast(file_path, output_path, solc_version):
    # Install the required version if not installed
    solc_install_cmd = f'solc-select install {solc_version}'
    subprocess.run(solc_install_cmd, shell=True, check=True)

    # Use the required version
    solc_select_cmd = f'solc-select use {solc_version}'
    subprocess.run(solc_select_cmd, shell=True, check=True)

    # Run solc to generate the AST JSON
    solc_ast_cmd = f'solc --ast-json {file_path} -o {output_path}'
    result = subprocess.run(solc_ast_cmd, shell=True, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        print(f'Error generating AST JSON for {file_path} with solc {solc_version}:')
        print(result.stderr)

        # Run solc to generate the AST JSON
        solc_ast_cmd = f'solc --ast-compact-json {file_path} -o {output_path}'
        result = subprocess.run(solc_ast_cmd, shell=True, stderr=subprocess.PIPE, text=True)

        if result.returncode != 0:
            print(f'Error generating Compact AST JSON for {file_path} with solc {solc_version}:')
            print(result.stderr)
        else:
            print(f'Processed {file_path} with solc {solc_version}')

        pass
    else:
        print(f'Processed {file_path} with solc {solc_version}')

def find_solc_version_from_code(sol_code):
    pragma_match = re.search(r'pragma solidity (.*);', sol_code)
    if pragma_match:
        version_range = pragma_match.group(1)
        if '>=' in version_range and '<' in version_range:
            min_version = re.search(r'>=(\d+\.\d+(\.\d+)?)', version_range)
            max_version = re.search(r'<(\d+\.\d+(\.\d+)?)', version_range)
            if min_version and max_version:
                # Use the minimum version from the range
                return min_version.group(1)
        elif '>=' in version_range:
            min_version = re.search(r'>=(\d+\.\d+(\.\d+)?)', version_range)
            if min_version:
                # Use the minimum version from the range
                return min_version.group(1)
        elif version_range.startswith('^'):
            # Extract specific version from range or use a default version
            specific_version = re.search(r'\^(\d+\.\d+(\.\d+)?)', version_range)
            if specific_version:
                specific_version = specific_version.group(1)
            else:
                specific_version = '0.8.0'

            # Format the version string correctly
            specific_version_parts = specific_version.split('.')
            if len(specific_version_parts) == 3 and specific_version_parts[2].startswith('0'):
                specific_version_parts[2] = re.sub(r'0+','0',specific_version_parts[2])
            formatted_version = '.'.join(specific_version_parts)
            return formatted_version
        return version_range
    return None

def find_solc_version(file_path):
    with open(file_path, 'r') as f:
        try:
            content = f.read()
        except:
            return None
        pragma_match = re.search(r'pragma solidity (.*);', content)
        if pragma_match:
            version_range = pragma_match.group(1)
            if '>=' in version_range and '<' in version_range:
                min_version = re.search(r'>=(\d+\.\d+(\.\d+)?)', version_range)
                max_version = re.search(r'<(\d+\.\d+(\.\d+)?)', version_range)
                if min_version and max_version:
                    # Use the minimum version from the range
                    return min_version.group(1)
            elif '>=' in version_range:
                min_version = re.search(r'>=(\d+\.\d+(\.\d+)?)', version_range)
                if min_version:
                    # Use the minimum version from the range
                    return min_version.group(1)
            elif version_range.startswith('^'):
                # Extract specific version from range or use a default version
                specific_version = re.search(r'\^(\d+\.\d+(\.\d+)?)', version_range)
                if specific_version:
                    specific_version = specific_version.group(1)
                else:
                    specific_version = '0.8.0'

                # Format the version string correctly
                specific_version_parts = specific_version.split('.')
                if len(specific_version_parts) == 3 and specific_version_parts[2].startswith('0'):
                    specific_version_parts[2] = re.sub(r'0+','0',specific_version_parts[2])
                formatted_version = '.'.join(specific_version_parts)
                return formatted_version
            return version_range
    return None




def process_directory_ast_json(input_dir, output_dir):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.endswith('.sol'):
                file_path = os.path.join(root, file)
                rel_path = os.path.relpath(root, input_dir)
                output_path = os.path.join(output_dir, rel_path)

                os.makedirs(output_path, exist_ok=True)

                json_output_file = os.path.join(output_path, file.replace('.sol', '.sol_json.ast'))

                #if 'arithmetic' not in json_output_file and 'access' not in json_output_file:
                #    continue

                if os.path.exists(json_output_file):
                    print(f'Skipped {file_path} as the .json file already exists')
                else:
                    solc_version = find_solc_version(file_path)
                    if solc_version:
                        try:
                            run_solc_ast(file_path, output_path, solc_version)
                            print(f'Processed {file_path} with solc {solc_version}')
                        except Exception as e:
                            print(f'Error processing {file_path}: {e}')
                    else:
                        print(f'Could not find solc version for {file_path}')

def extract_text_from_pdfs(dir_path):
    """
    Extracts text from all PDF files in the specified directory and saves it in a .md format.
    :param dir_path: The path to the directory containing the PDF files.
    """
    # Get a list of all PDF files in the directory
    pdf_files = [f for f in os.listdir(dir_path) if f.endswith('.pdf')]
    
    # Iterate over each PDF file and extract its text
    for pdf_file in pdf_files:
        with open(os.path.join(dir_path, pdf_file), 'rb') as f:
            pdf_reader = PyPDF2.PdfReader(f)
            text = ''
            for page in pdf_reader.pages:
                text += page.extract_text()
            
            # Write the extracted text to a .md file with the same name as the PDF file
            with open(os.path.join(dir_path, pdf_file[:-4] + '.md'), 'w') as f:
                f.write(text)

class Node:

    def __init__(self, json_node, id):
        self.id = id
        self.json_node = json_node
        self.type = json_node['name']
        self.control_edges = []
        self.sequential_edges = []
        self.data_edges = []

    def __str__(self):
        node_type = self.json_node.get("nodeType", "")
        name = self.json_node.get("name")
        value = self.json_node.get("attributes", {}).get("value") or self.json_node.get("attributes", {}).get("operator")
        details = []

        

        if name:
            details.append(f"{name}")
        
        # try:
        #     for attr_key in self.json_node.get('attributes').keys():
        #         details.append("A="+attr_key+' '+self.json_node.get('attributes')[attr_key])
        # except:
        #     pass

        # if value:
        #     details.append(f"V={value}")

        if "type" in self.json_node:
            details.append(f"{self.json_node['type']}")

        details_str = " ".join(details)
        return f"{self.id}"
        #return f"{self.id} {node_type} {details_str}"

class ASG:

    SOLIDITY_TYPES = {
        'bool': 0,
        'address': 1,
        'int': 2,
        'uint': 3,
        'fixed': 4,
        'ufixed': 5,
        'bytes': 6,
        'string': 7,
        'mapping': 8,
        'function': 9,
        'struct': 10
    }

    NODE_TYPES = {
        'VariableDeclaration': 0,
        'Assignment': 1,
        'FunctionCall': 2,
        'Block': 3,
        'IfStatement': 4,
        'WhileStatement': 5,
        'DoWhileStatement': 6,
        'ForStatement': 7
    }

    @staticmethod
    def one_hot_encode_solidity_type(solidity_type):
        """
        Returns a one-hot encoded vector for the given Solidity variable type.

        Args:
        solidity_type (str): The Solidity variable type as a string.

        Returns:
        np.ndarray: A one-hot encoded vector.
        """
        solidity_type = re.sub(r'\d+','',solidity_type)
        solidity_type = solidity_type.split(" ")[0]

        num_types = len(ASG.SOLIDITY_TYPES)
        
        # If the input type is not in the mapping, raise an error
        if solidity_type not in ASG.SOLIDITY_TYPES:
            #raise ValueError(f"Unknown Solidity type: {solidity_type}")
            one_hot_vector = np.zeros(num_types, dtype=np.float32)
            return one_hot_vector
        
        # Create an empty one-hot encoding vector
        one_hot_vector = np.zeros(num_types, dtype=np.float32)
        
        # Set the corresponding element in the vector to 1
        index = ASG.SOLIDITY_TYPES[solidity_type]
        one_hot_vector[index] = 1.0
        
        return one_hot_vector

    @staticmethod
    def one_hot_encode_solidity_node(node_type):
        """
        Returns a one-hot encoded vector for the given Solidity node type.

        Args:
        node_type (str): The graph node type as a string.

        Returns:
        np.ndarray: A one-hot encoded vector.
        """
        node_type = re.sub(r'\d+','',node_type)

        num_types = len(ASG.NODE_TYPES)
        
        # If the input type is not in the mapping, raise an error
        if node_type not in ASG.NODE_TYPES:
            raise ValueError(f"Unknown Solidity type: {node_type}")
        
        # Create an empty one-hot encoding vector
        one_hot_vector = np.zeros(num_types, dtype=np.float32)
        
        # Set the corresponding element in the vector to 1
        index = ASG.NODE_TYPES[node_type]
        one_hot_vector[index] = 1.0
        
        return one_hot_vector

    def __init__(self, solidity_ast_json):
        self.ast = json.loads(solidity_ast_json)
        self.nodes = []
        node_counter = 0 
        for node in self.walk(self.ast):
            new_node = Node(node,node_counter)
            node_counter += 1
            self.nodes.append(new_node)
        self.variable_set = set()

    def walk(self, node):
        if isinstance(node, dict):
            if 'name' in node and node['name'] in ("VariableDeclaration", "Assignment", "FunctionCall", "Block", "ElseStatement", "IfStatement", "WhileStatement", "DoWhileStatement", "ForStatement"):
                #yield from self.walk(value)
                yield node
            for key, value in node.items():
                if isinstance(value, (dict, list)):
                    yield from self.walk(value)
        elif isinstance(node, list):
            for item in node:
                yield from self.walk(item)

    def generate(self):
        self.add_control_and_sequential_edges()
        self.add_data_edges()
        return self

    def add_control_and_sequential_edges(self):
        for i in range(len(self.nodes) - 1):
            current_node = self.nodes[i]
            next_node = self.nodes[i + 1]

            if current_node.json_node['name'] in ("FunctionCall", "Block", "IfStatement", "WhileStatement", "DoWhileStatement", "ForStatement"):
                one_hot_encoded_vector = ASG.one_hot_encode_solidity_node(current_node.json_node['name'])
                current_node.control_edges.append([next_node, one_hot_encoded_vector])
            else:
                one_hot_encoded_vector = ASG.one_hot_encode_solidity_node(current_node.json_node['name'])
                current_node.sequential_edges.append([next_node, one_hot_encoded_vector])
                pass

            if current_node.json_node['name'] in ("VariableDeclaration", "Assignment"):
                if "attributes" in current_node.json_node:
                    attributes = current_node.json_node["attributes"]
                    if "name" in attributes and "type" in attributes:
                            self.variable_set.add(attributes['type']+'#'+attributes['name'])
            
    def add_data_edges(self):
        for var in self.variable_set:
            var_type = var.split("#")[0]
            var_value = var.split("#")[1]
            nodes_with_var = [node for node in self.nodes if self.contains_var(node, var)]
            for i in range(len(nodes_with_var) - 1):
                one_hot_encoded_vector = ASG.one_hot_encode_solidity_type(var_type)
                if (len(one_hot_encoded_vector)<10):
                    one_hot_encoded_vector += 10-len(one_hot_encoded_vector) * [0]
                    print(one_hot_encoded_vector)
                nodes_with_var[i].data_edges.append([nodes_with_var[i + 1], one_hot_encoded_vector])

    def contains_var(self, node, var):
            var_type = var.split("#")[0]
            var_value = var.split("#")[1]
            for subnode in self.walk(node.json_node):
                if isinstance(subnode, dict) and 'name' in subnode and subnode['name'] in ("VariableDeclaration", "Assignment"):
                    if subnode['name']=='Assignment':
                        for assign_attr in subnode['children']:
                            if 'attributes' in assign_attr and 'value' in assign_attr['attributes']  and assign_attr['attributes']['type'] == var_type and assign_attr['attributes']['value'] == var_value:
                                    return True
                    elif subnode['name']=='VariableDeclaration':
                        if 'attributes' in subnode and 'value' in subnode['attributes']  and subnode['attributes']['type'] == var_type and (subnode['attributes']['name'] == var_value):
                                return True
            return False
    
    def to_networkx_graph(self):
        g = nx.DiGraph()

        for node in self.nodes:
            g.add_node(node, features=ASG.one_hot_encode_solidity_node(node.type))

        for node in self.nodes:
            for succ in node.control_edges:
                g.add_edge(node.id, succ[0].id, label="C", features=succ[1])
            for succ in node.sequential_edges:
                g.add_edge(node.id, succ[0].id, label="S", features=succ[1])
            for succ in node.data_edges:
                g.add_edge(node.id, succ[0].id, label="D", features=succ[1])

        return g

    def draw(self):
        g = self.to_networkx_graph()
        pos = nx.spring_layout(g, seed=42)
        nx.draw(g, pos, with_labels=True, node_color="lightblue", font_size=8, font_weight="bold")
        edge_labels = nx.get_edge_attributes(g, 'label')
        nx.draw_networkx_edge_labels(g, pos, edge_labels=edge_labels, font_size=8)
        plt.show()

def _pack_batch(graphs):
        """Pack a batch of graphs into a single `GraphData` instance.
        Args:
        graphs: a list of generated networkx graphs.
        Returns:
        graph_data: a `GraphData` instance, with node and edge indices properly
            shifted.
        """
        # Graphs = []
        # for graph in graphs:
        #     for inergraph in graph:
        #         Graphs.append(inergraph)
        # graphs = Graphs
        from_idx = []
        to_idx = []
        graph_idx = []

        n_total_nodes = 0
        n_total_edges = 0
        g_i_node_features = {}
        g_i_edge_features = {}
        stacked_node_features = []
        stacked_edge_features = []

        for i, g in enumerate(graphs):
            g_i_node_features = {}
            g_i_edge_features = {}

            for node in g.nodes():
                if type(node)!=int and type(node)!=np.int64:
                    if len(nx.get_node_attributes(g, 'features').keys())>0:
                        g_i_node_features[node] = nx.get_node_attributes(g, 'features')[node]
                        stacked_node_features.append(g_i_node_features[node])
                    else:
                        g_i_node_features[node] = [0] * 8
                        stacked_node_features.append(g_i_node_features[node])
                else:
                    g_i_node_features[node] = [0] * 8
                    stacked_node_features.append(g_i_node_features[node])

            for u, v, data in g.edges(data=True):
                try:
                    g_i_edge_features[str(u) + '#' + str(v)] = data['features']
                    padded_features = np.append(g_i_edge_features[str(u) + '#' + str(v)] , [0] * (10 - len(g_i_edge_features[str(u) + '#' + str(v)])))
                    stacked_edge_features.append(padded_features)
                except KeyError:
                    g_i_edge_features[str(u) + '#' + str(v)] = [0] * 10
                    stacked_edge_features.append(g_i_edge_features[str(u) + '#' + str(v)])

            n_nodes = len(g_i_node_features) or 1
            n_edges = len(g_i_edge_features) or 1
            edge_indices = np.array([[u, v] for u, v in g.edges()], dtype=np.int32)

            if len(edge_indices)<1:
                #from_idx.append([0]*n_total_nodes)
                #to_idx.append([0]*n_total_nodes)
                continue
            else:
                from_idx.append(edge_indices[:, 0] + n_total_nodes)
                to_idx.append(edge_indices[:, 1] + n_total_nodes)
            graph_idx.append(np.ones(n_nodes, dtype=np.int32) * i)

            n_total_nodes += n_nodes
            n_total_edges += n_edges

        GraphData = collections.namedtuple('GraphData', [
            'from_idx',
            'to_idx',
            'node_features',
            'edge_features',
            'graph_idx',
            'n_graphs'])            
        
        min_node_features = min([len(x) for x in stacked_node_features])
        stacked_node_features = [x[:min_node_features] for x in np.array(stacked_node_features)]

        min_edge_features = min([len(x) for x in stacked_edge_features])
        stacked_edge_features = [x[:min_edge_features] for x in np.array(stacked_edge_features)]

        stacked_node_features = np.array(stacked_node_features, dtype=np.float32)
        stacked_edge_features = np.array(stacked_edge_features, dtype=np.float32)
        return GraphData(
            from_idx=np.concatenate(from_idx, axis=0),
            to_idx=np.concatenate(to_idx, axis=0),
            # this task only cares about the structures, the graphs have no features.
            # setting higher dimension of ones to confirm code functioning
            # with high dimensional features.
            #node_features=np.ones((n_total_nodes, 8), dtype=np.float32),
            #edge_features=np.ones((n_total_edges, 4), dtype=np.float32),
            node_features=stacked_node_features,
            edge_features=stacked_edge_features,
            graph_idx=np.concatenate(graph_idx, axis=0),
            n_graphs=len(graphs),
        )