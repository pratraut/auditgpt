import os
import re
import csv
import glob

def remove_hyperlinks(text):
    # remove hyperlinks that start with http or https
    text = re.sub(r'http\S+|https\S+', '[LINK]', text)
    
    # remove hyperlinks that start with www
    text = re.sub(r'www.\S+', '[LINK]', text)
    
    # remove hyperlinks that include "#"
    #text = re.sub(r'#\S+', '', text)
    
    return text

def parse_md_report(issue, text_file, issues_dir):


    # regex pattern to match links to relevant lines of code
    pattern = r'(https:\/\/github\.com\/\S+\/\S+\/blob\/\S+\/\S+#L\d+)(-#L(\d+))?'

    # extract the links and file information
    matches = re.findall(pattern, issue)

    if len(matches)<1:
        return None
    
    m_set = list()
    for m in matches:
        m_set.append(m[0])
    m_set = list(set(m_set))
    
    for match in m_set:
        for split_match in match.split("\\n"):
            try:
                url = split_match.replace("/blob/","/raw/")
            except:
                continue
            link = split_match.strip()        
            filename = link.split('/')[-1]
            source_filename = filename.split("#")[0]
            line_range = link.split('#')[-1].replace(")","")
            if '-' in line_range:
                try:
                    start_line, end_line = line_range.strip('-').split('-')
                except:
                    start_line = end_line = line_range.split('-')[0]
            else:
                start_line = end_line = line_range

            # Find the corresponding _code.txt file
            text_prefix = '_'.join(text_file.split("_md_")[0].split('_')[:-1])
            issue_id = 0
            md_code_files = glob.glob(f"{os.getenv('HOME_DIR')}/data/{issues_dir}/{text_prefix}_{source_filename}_*_{issue_id}_code_md.txt")
            if len(md_code_files)<1:
                continue
            for code_file in md_code_files:
                
                if os.path.exists(os.path.join(folder_path, code_file)):

                    # Open the code file and extract the lines corresponding to the line number range
                    with open(os.path.join(folder_path, code_file), 'r') as f:
                        code_lines = f.readlines()
                    
                    temp_content = ' '.join(code_lines)
                    if '<tr>' in temp_content:
                        continue
                    if '<link>' in temp_content:
                        continue
                    if '<li>' in temp_content:
                        continue

                    try:
                        if '//' in code_lines[int(start_line.replace("L",''))].strip()[:5] or '*' in code_lines[int(start_line.replace("L",''))].strip()[:5]:
                            continue
                    except:
                        continue
                                        
                    start_code = int(start_line.replace("L",''))
                    end_code = int(end_line.replace("L",''))+1
                    for i in range(int(end_line.replace("L",'')), 10000, 1):
                        try:
                            if ('function ' in code_lines[i] or 'constructor(' in code_lines[i] or 'destroy(' in code_lines[i]) and ('//' not in code_lines[i].strip()[:5] and '*' not in code_lines[i].strip()[:5]):
                                end_code = i
                                break
                        except:
                            pass
                    for i in range(int(start_line.replace("L",'')), 0, -1):
                        try:
                            if ('function ' in code_lines[i] or 'constructor(' in code_lines[i] or 'destroy(' in code_lines[i]) and ('//' not in code_lines[i].strip()[:5] and '*' not in code_lines[i].strip()[:5]):
                                start_code = i
                                break
                        except:
                            pass

                    # Compile the Solidity code and extract the function containing the code lines
                    source = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', '\n'.join(code_lines[start_code:end_code])).strip()
                    source = re.sub(r'\t+',' ',source.replace('\n', '').replace('\r',' ').replace('\t', ' '))
                    
                    code_stripped = re.sub(r'\s+',' ',source.replace('\n', ' ').replace('\r',' ').replace('\t', ' '))

                    content_issue = "\n".join(issue.split("\n"))
                    content_issue = re.sub(r'\t+',' ',content_issue)
                    content_issue = remove_hyperlinks(content_issue)
                    content_issue = re.sub(r'\n+',' [n] ',content_issue)

                    #print(text_file)
                    fcsv = open(f"{os.getenv('HOME_DIR')}/issues_{issues_dir}.csv", "a+")
                    writer = csv.writer(fcsv, delimiter='\t')

                    # write the rows to the CSV file
                    writer.writerow([text_file.split("_text")[0], code_stripped, content_issue])
            
        

def parse_vuln_report(content, issues_dir):
    

    # regex pattern to match links to relevant lines of code
    pattern = r'(https:\/\/github\.com\/\S+\/\S+\/blob\/\S+\/\S+#L\d+)(-#L(\d+))?' #r'(https:\/\/github\.com\/.+\/(.+\.sol)#L(\d+)(-L(\d+))?)\b'
    
    # extract the links and file information
    matches = re.findall(pattern, content)

    if len(matches)<1:
        return None
    
    m_set = list()
    for m in matches:
        m_set.append(m[0])
    m_set = list(set(m_set))

    for match in m_set:
        for split_match in match.split("\\n"):
            try:
                url = split_match.replace("/blob/","/raw/")
            except:
                continue
            link = split_match.strip()
            filename = link.split('/')[-1]
            source_filename = filename.split("#")[0]
            line_range = link.split('#')[-1].replace(")","")
            if '-' in line_range:
                try:
                    start_line, end_line = line_range.strip('-').split('-')
                except:
                    start_line = end_line = line_range.split('-')[0]
            else:
                start_line = end_line = line_range

            # Find the corresponding _code.txt file
            text_prefix = text_file.split("_vuln_")[0]
            vuln_code_files = glob.glob(f"{os.getenv('HOME_DIR')}/data/{issues_dir}/{text_prefix}_{source_filename}_*_code_vuln.txt")
            if len(vuln_code_files)<1:
                continue

            for code_file in vuln_code_files:
                if os.path.exists(os.path.join(folder_path, code_file)):
                    # Open the code file and extract the lines corresponding to the line number range
                    with open(os.path.join(folder_path, code_file), 'r') as f:
                        code_lines = f.readlines()
                    
                    temp_content = ' '.join(code_lines)
                    if '<tr>' in temp_content:
                        continue
                    if '<link>' in temp_content:
                        continue
                    if '<li>' in temp_content:
                        continue

                    try:
                        if '//' in code_lines[int(start_line.replace("L",''))].strip()[:5] or '*' in code_lines[int(start_line.replace("L",''))].strip()[:5]:
                            continue
                    except:
                        continue
                                            
                    start_code = int(start_line.replace("L",''))
                    end_code = int(end_line.replace("L",''))+1
                    for i in range(int(end_line.replace("L",'')), 10000, 1):
                        try:
                            if ('function ' in code_lines[i] or 'constructor(' in code_lines[i] or 'destroy(' in code_lines[i]) and ('//' not in code_lines[i].strip().strip()[:5] and '*' not in code_lines[i].strip().strip()[:5]):
                                end_code = i
                                break
                        except:
                            pass
                    for i in range(int(start_line.replace("L",'')), 0, -1):
                        try:
                            if ('function ' in code_lines[i] or 'constructor(' in code_lines[i] or 'destroy(' in code_lines[i]) and ('//' not in code_lines[i].strip().strip()[:5] and '*' not in code_lines[i].strip().strip()[:5]):
                                start_code = i
                                break
                        except:
                            pass

                    # Compile the Solidity code and extract the function containing the code lines
                    source = re.sub(r'\/\/.*|\/\*[\s\S]*?\*\/', '', '\n'.join(code_lines[start_code:end_code])).strip()
                    source = re.sub(r'\t+',' ',source.replace('\n', '').replace('\r',' ').replace('\t', ' '))
                    
                    code_stripped = re.sub(r'\s+',' ',source.replace('\n', ' ').replace('\r',' ').replace('\t', ' '))

                    content_vuln = "\n".join(content.split("\n"))
                    content_vuln = re.sub(r'\t+',' ',content_vuln)
                    content_vuln = remove_hyperlinks(content_vuln)
                    content_vuln = re.sub(r'\n+',' [n] ',content_vuln)

                    #print(text_file)
                    fcsv = open(f"{os.getenv('HOME_DIR')}/issues_{issues_dir}.csv", "a+")
                    writer = csv.writer(fcsv, delimiter='\t')

                    # write the rows to the CSV file
                    writer.writerow([text_file.split("_text")[0], code_stripped, content_vuln])

issues_dirs = ['repos_data_code-423n4_issues','repos_data_sherlock-audit_issues']
for issues_dir in issues_dirs:
    folder_path = f'{os.getenv("HOME_DIR")}/data/{issues_dir}'

    filecount = 0
    total_files = sum([len(files) for r, d, files in os.walk(folder_path)])

    # Recursively walk through the directory
    for root, dirs, files in os.walk(folder_path):
        for text_file in files:
            print(f"{filecount} / {total_files}")
            if text_file.endswith('_text.txt'):
                file_path = os.path.join(root, text_file)
                with open(file_path, 'r') as f:
                    content = f.read()

                if '_vuln_text' in text_file:
                    parse_vuln_report(content, issues_dir)

                if '_md_text' in text_file:
                    parse_md_report(content, text_file, issues_dir)
            filecount+=1
                