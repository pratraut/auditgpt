import pickle
import random
import sys
import string
from datetime import datetime

import torch

from audit_utils import *
from audit_utils import _pack_batch

from gmn.evaluation import compute_similarity
from gmn.utils import load_pickles_from_directory, reshape_and_split_tensor
from gmn.configure import get_default_config

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
torch.autograd.set_detect_anomaly(True)
sys.setrecursionlimit(10000) 

# tokenizer = AutoTokenizer.from_pretrained("microsoft/CodeBERT")
# model = AutoModelForMaskedLM.from_pretrained("microsoft/CodeBERT")
# scorer = BERTScorer(model_type="microsoft/CodeBERT")

import torch


def generate_report(sol_code, base_path):

    retObj={}
    retObj['graphs']={}

    # Get the current datetime and format it as a timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    # Count the number of lines in the text
    line_count = len(sol_code.splitlines())

    # Generate a random 6-character hash
    random_hash = ''.join(random.choices(string.ascii_letters + string.digits, k=6))

    # Create the filename with the timestamp and line count
    filename = f'code_{timestamp}_{line_count}_{random_hash}'
    filepath_code = f"{base_path}/{filename}.txt"
    filepath_json = f"./user_data/json/{filename}.txt_json.ast"
    filedir_json = f"./user_data/json"
    filepath_pickle = f"./user_data/pickle/{filename}.pickle"

    #sol_code = re.sub(r'(import\s+[^;]+;)', r'// \1', sol_code)
    # Define a regular expression pattern to match Hardhat import statements
    hardhat_import_pattern = r'^import\s+"hardhat/.*?";\s*$'

    # Replace Hardhat import statements with an empty string
    sol_code = re.sub(hardhat_import_pattern, "", sol_code, flags=re.MULTILINE)


    # Write the result to a file
    with open(filepath_code, "w+") as code_file:
        code_file.write(sol_code)

    solc_version = find_solc_version(filepath_code)
    if solc_version:
        try:
            run_solc_ast(filepath_code, filedir_json, solc_version)
            print(f'Processed {filepath_code} with solc {solc_version}')
        except Exception as e:
            print(f'Error processing {filepath_code}: {e}')
    else:
        print(f'Could not find solc version for {filepath_code}')

    with open(filepath_json, 'r') as f:
        solidity_ast_json = f.read()

    try:
        asg = ASG(solidity_ast_json)
        asg.generate()
        graph = asg.to_networkx_graph()
        
        with open(filepath_pickle, 'wb') as pf:
            pickle.dump(graph, pf)

        print(f'Successfully converted and saved {filepath_json} as a pickle in {filepath_pickle}')
    except Exception as e:
        print(f'Error converting {filepath_json} to a networkx Graph: {e}')


    # Print configure
    config = get_default_config()

    #config = copy.deepcopy(config)

    graph_types = os.listdir(f'{os.getenv("HOME_DIR")}/models')
    for gt in graph_types:
        pickle_directory = f'{os.getenv("HOME_DIR")}/data/known_vulnerabilities_pickle/{gt}'
        loaded_graphs = load_pickles_from_directory(pickle_directory)
        loaded_graphs.append(graph)
        loaded_graphs = loaded_graphs[-2:]

        graphData = _pack_batch(loaded_graphs)

        node_features = torch.from_numpy(graphData.node_features)
        edge_features = torch.from_numpy(graphData.edge_features)
        from_idx = torch.from_numpy(graphData.from_idx).long()
        to_idx = torch.from_numpy(graphData.to_idx).long()
        graph_idx = torch.from_numpy(graphData.graph_idx).long()

        node_feature_dim = graphData.node_features.shape[-1]
        edge_feature_dim = graphData.edge_features.shape[-1]

        model_list = os.listdir(f'{os.getenv("HOME_DIR")}/models/{gt}')
        highest_auc_model = [(idx, float(w.split("_")[-4].replace(".pth","")), w) for idx, w in enumerate(model_list)]
        highest_auc_model = sorted(highest_auc_model, key=lambda x: x[1])[-1]
        
        edge_feature_dim = int(highest_auc_model[2].split('_')[-1].replace(".pth",''))
        node_feature_dim = int(highest_auc_model[2].split('_')[-2])

        #GMN_Model, optimizer = gmn.utils.build_model(config, node_feature_dim, edge_feature_dim)

        with open(f"{os.getenv('HOME_DIR')}/models/{gt}/{highest_auc_model[2]}", 'rb') as f:
             GMN_Model = pickle.load(f)

        weight_list = os.listdir(f'{os.getenv("HOME_DIR")}/weights/{gt}')
        lowest_loss_weight = [(idx, float(w.split("_")[-3].replace(".pth","")), w) for idx, w in enumerate(weight_list)]
        lowest_loss_weight = sorted(lowest_loss_weight, key=lambda x: x[1])

        # Load the model weights
        GMN_Model.load_state_dict(torch.load(f"{os.getenv('HOME_DIR')}/weights/{gt}/{lowest_loss_weight[0][2]}"))

        # Pass the data through the GMN_Model
        with torch.no_grad():
            GMN_Model.eval()
            graph_vectors = GMN_Model(node_features, edge_features, from_idx, to_idx, graph_idx, 2)
            x, y = reshape_and_split_tensor(graph_vectors, 2)
            similarity_score = compute_similarity(config, x, y)
            retObj['graphs'][gt] = float(similarity_score.detach().numpy()[0])

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

def main():
    directory = f'{os.getenv("HOME_DIR")}/test'
    sol_files = []
    target_filename = directory+'/report_graphs.md'

    collect_sol_files(directory, sol_files)

    base_path = os.path.commonpath(sol_files)
    print('List of Solidity (.sol) files:')
    for sol_file in sol_files:
        print(sol_file)
        with open(sol_file, 'r', encoding='utf-8') as file:
            content = file.read()
            contract_obj = generate_report(content, base_path)

        for f_name in contract_obj['graphs']:
            function_list = contract_obj['graphs'][f_name]
            if len(function_list)>0:
                with open(target_filename, 'a+', encoding='utf-8') as file:
                    file.write(f"{function_list[0]['title']}:\n")
                    for f_obj in function_list:
                        file.write(f"  -- {f_obj['explanation']}\n")

if __name__ == '__main__':
    main()