# AuditGPT

AuditGPT is a state-of-the-art tool designed to enhance the security of Solidity smart contracts. Utilizing the power of GPT-3.5 and GPT-4 models, this project scrapes known vulnerabilities and scans Solidity smart contracts for potential threats.

[![AuditGPT Introduction](http://img.youtube.com/vi/vYEu9cSsIs8/0.jpg)](https://www.youtube.com/watch?v=vYEu9cSsIs8 "AuditGPT Introduction")


If similarities are found, they are sent along with the code to the AI model to improve the results of an audit. The primary goal is to identify and mitigate potential security risks before they become an issue.

## Features

- Scrapes and learns from known vulnerabilities
- Scans Solidity smart contracts for potential risks
- Uses GPT-3.5 and GPT-4 models to enhance audit results
- Provides detailed audit report with potential vulnerabilities and mitigation strategies

## Requirements

- Python 3.9 or newer
- OpenAI API key

## Installation

\`\`\`bash
git clone https://github.com/mkondov/auditgpt.git
cd AuditGPT
pip install -r requirements.txt
\`\`\`

## Usage

To use AuditGPT, follow these steps:

1. Set your OpenAI API key, Github API key and home folder in the .env file (rename example.env to .env)
2. Run clone_repos.sh or github_scraper.py # this downloads sherlock and code4arena's repositories and issues
3. Run gather_sol.py # this generates matching issue-code pairs from the repos
4. Run compile_to_csv.py # this reads the issue-code pairs and joins them in a single .csv
5. Run prepare_df.py # creates a pickled Pandas dataframe with encoded code-explanation pairs
---- Those steps have to be done only once

6. Put all .sol contracts you want analyzed in /test folder and run

\`\`\`bash
python gen_report_functions.py
\`\`\`

AuditGPT will output a report detailing any potential vulnerabilities found and provide suggestions for improving the security of your smart contract.

## Contributing

We welcome contributions from the community! Please read our [Contributing Guide](CONTRIBUTING.md) to get started.

## License

AuditGPT is released under the GNU GPL v3.0 License. See the [LICENSE](LICENSE.md) file for more details.

## Disclaimer

While AuditGPT strives to provide accurate and helpful information, it does not guarantee the security of your smart contracts. The tool should be used as a part of a broader security strategy. Always seek professional advice when dealing with critical smart contracts.

## Contact

If you have any questions, issues, or suggestions, please file an issue in this repository or contact me at martin.kondov@gmail.com

## Acknowledgments

This project uses code from the OpenAI API, which is copyright OpenAI.
