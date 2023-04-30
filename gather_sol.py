import os
import re
import json
import glob
import requests

def read_md_files_with_numbers(root_dir):
    md_files_with_numbers = []

    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.md') and re.search(r'\d', file):
                md_files_with_numbers.append((file, subdir))

    return md_files_with_numbers

def gather_json_data(root_dir):
    json_data = []
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(subdir, file)) as f:
                    try:
                        data = json.load(f)
                        if 'issueUrl' not in data.keys():
                            continue
                        #print(data)
                        json_data.append([data, subdir])
                    except:
                        pass
    return json_data

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

def get_markdown_issue_content(issue_url, auth_token):
    url = issue_url.replace("/blob/","/raw/")
    if '.md' not in url:
        return None
    headers = {'Authorization': f'token {auth_token}'}
    response = requests.get(url, headers = headers)
    if response.status_code != 200:
        return None
    content = str(response.content)
    return content

def parse_markdown_report(issue_body, auth_token, jfile):

    #issue_body = re.sub(r'#L','[STAR]L',issue_body)
    issue_body_sectioned = re.sub(r'#+(\s+)?\d(\d)?', '[SECTION]', str(issue_body))
    issues = issue_body_sectioned.split("[SECTION]")
    if len(issues)<2:
        return None
    
    issues = issues[1:]

    downloaded_sols = []
    for issue_id, issue in enumerate(issues):

        # regex pattern to match links to relevant lines of code
        pattern = r'(https:\/\/github\.com\/\S+\/\S+\/blob\/\S+\/\S+#L\d+)(-#L(\d+))?'
        sub_pattern = r'(https:\/\/github\.com\/.+\/(.+\.sol)#L(\d+)(-L(\d+))?)\b'

        # extract the links and file information
        matches = re.findall(pattern, issue)

        if len(matches)<1:
            continue

        matches = matches[:3]

        headers = {'Authorization': f'token {auth_token}'}
        
        # use Github API to retrieve the file content
        for match_id, match in enumerate(matches):
            for split_match_id, split_match in enumerate(match[0].split("\\n")):
                try:
                    url = split_match.replace("/blob/","/raw/")
                except:
                    continue
                if '.sol' not in url:
                    continue

                sub_matches = re.findall(sub_pattern, url)
                if len(sub_matches)<1:
                    continue
                sub_match = sub_matches[0]

                if sub_match[1] in downloaded_sols:
                    continue

                response = requests.get(sub_match[0], headers = headers)
                if response.status_code == 200:
                    content = str(response.content)
                    function_lines = content.split('\\n')
                    function_code = '\n'.join(function_lines)
                    with open(f"{os.getenv('HOME_DIR')}/data/{dst_dir}/{subdir.split('/')[-2]}__{jfile['issueId']}_{jfile['handle']}_{sub_match[1]}_{split_match_id}_{match_id}_{issue_id}_code_md.txt", "w+") as f:
                        f.write(function_code)
                    downloaded_sols.append(sub_match[1])
        with open(f"{os.getenv('HOME_DIR')}/data/{dst_dir}/{subdir.split('/')[-2]}__{jfile['issueId']}_{jfile['handle']}_{issue_id}_md_text.txt", "w+") as f:
            f.write(issue)

def parse_vulnerability_report(issue_body, auth_token, jfile):


    # regex pattern to match links to relevant lines of code
    pattern = r'(https:\/\/github\.com\/\S+\/\S+\/blob\/\S+\/\S+#L\d+)(-#L(\d+))?'
    sub_pattern = r'(https:\/\/github\.com\/.+\/(.+\.sol)#L(\d+)(-L(\d+))?)\b'

    # extract the links and file information
    matches = re.findall(pattern, issue_body)

    if len(matches)<1:
        return None
    
    matches = matches[:3]

    headers = {'Authorization': f'token {auth_token}'}
    downloaded_sols = []
    # use Github API to retrieve the file content
    for match_id, match in enumerate(matches):
        for split_match_id, split_match in enumerate(match[0].split("\\n")):
            try:
                url = split_match.replace("/blob/","/raw/")
            except:
                continue
            if '.sol' not in url:
                continue
            sub_matches = re.findall(sub_pattern, url)
            if len(sub_matches)<1:
                continue
            sub_match = sub_matches[0]

            if sub_match[1] in downloaded_sols:
                continue
            
            response = requests.get(sub_match[0], headers = headers)
            if response.status_code == 200:
                content = str(response.content)
                function_lines = content.split('\\n')
                function_code = '\n'.join(function_lines)
                with open(f"{os.getenv('HOME_DIR')}/data/{dst_dir}/{competition_name}__{jfile['issueId']}_{jfile['handle']}_{sub_match[1]}_{split_match_id}_{match_id}_code_vuln.txt", "w+") as f:
                    f.write(function_code)
                downloaded_sols.append(sub_match[1])
    with open(f"{os.getenv('HOME_DIR')}/data/{dst_dir}/{competition_name}__{jfile['issueId']}_{jfile['handle']}_vuln_text.txt", "w+") as f:
        f.write(issue_body)


auth_token=os.getenv("GITHUB_ACCESS_TOKEN")

root_dirs = ['repos_data_code-423n4','repos_data_sherlock-audit',]
for root_dir in root_dirs:
    jcount = 0
    dst_dir = root_dir+"_issues"
    root_dir = f'{os.getenv("HOME_DIR")}/data/{root_dir}'

    try:
        os.makedirs(f'{os.getenv("HOME_DIR")}/data/{dst_dir}')
    except:
        pass
    
    if 'sherlock' in root_dir:
        md_files = read_md_files_with_numbers(root_dir)
        jtotal = len(md_files)
        for sfile, subdir in md_files:
            jcount += 1
            print(f"{jcount} / {jtotal} sherlock")
            try:
                if subdir.split('/')[-1][:2]!='20':
                    competition_name = subdir.split('/')[-2] + "_" + subdir.split('/')[-1]
                else:
                    competition_name = subdir.split('/')[-1]
                md_files = glob.glob(f"{os.getenv('HOME_DIR')}/data/{dst_dir}/{competition_name}__{sfile.replace('.md','')}*vuln*text.txt")
                if len(md_files)>0:
                    continue

                with open(subdir+'/'+sfile, 'r', encoding='utf-8') as file:
                    issue_body = file.read()

                if "vulnerability detail" in issue_body.lower():
                    jfile = {}
                    jfile['issueId'] = sfile.replace(".md","")
                    jfile['handle'] = 'sherlock'
                    parse_vulnerability_report(issue_body, auth_token, jfile)
            except Exception as err:
                print(err)

    elif 'code' in root_dir: #c4a
        json_files = gather_json_data(root_dir)
        jtotal = len(json_files)
        for jfile, subdir in json_files:
            jcount+=1
            print(f"{jcount} / {jtotal} c4a")
            try:

                competition_name = subdir.split('/')[-2]

                if os.path.exists(f"{os.getenv('HOME_DIR')}/data/{dst_dir}/{competition_name}__{jfile['issueId']}_{jfile['handle']}_vuln_text.txt"):
                    continue

                md_files = glob.glob(f"{os.getenv('HOME_DIR')}/data/{dst_dir}/{competition_name}__{jfile['issueId']}_{jfile['handle']}*md*text.txt")
                if len(md_files)>0:
                    continue

                issue_body = get_github_issue_content(jfile['issueUrl'], auth_token)
                issue_body = jfile['title'] + ' !!! \n\n' + issue_body

                if "vulnerability detail" in issue_body.lower():
                    parse_vulnerability_report(issue_body, auth_token, jfile)
                    continue

                markdown_regex = r'(https:\/\/github\.com\/.*?\.md)'
                md_matches = re.findall(markdown_regex, issue_body)

                if len(issue_body)<500 and len(md_matches)>0:
                    issue_body = get_markdown_issue_content(md_matches[0], auth_token)
                    parse_markdown_report(issue_body, auth_token, jfile)
                    continue
                
            except Exception as err:
                print(err)


