import logging
import os
import pandas as pd
import requests
import requests_cache

from datetime import datetime, date, timedelta
from git import Repo

import logging
import sys

from dotenv import load_dotenv

load_dotenv()

class GithubScraper():
	"""docstring for GithubScraper"""
	def __init__(self, console_handler, cache=''):
		super(GithubScraper, self).__init__()
		requests_cache.install_cache(cache, expire_after=timedelta(days=1)) # Cache repo data for one day (prevent reaching API rate limit)
		self.console_handler = console_handler
		self.base = f"https://api.github.com/"
		self.headers = {'Authorization': 'token ' + os.getenv('API_ACCESS_TOKEN'), 'User-Agent': 'Bot'} # Using auth bumps the rate limit to 5_000 requests per HOUR 

	def _check_request(self, req):
		if (req.status_code == 403 or req.status_code == 404):
			logging.critical(f"Request returned {req.status_code}: {req.json()}")
			exit(1)
		elif all(k in req.headers for k in ['x-ratelimit-limit', 'x-ratelimit-remaining']):
			logging.debug(f"Rate limit: {req.headers['x-ratelimit-remaining']} requests remaining (limit: {req.headers['x-ratelimit-limit']})")

		return req

	def _get_paginated(self, start_url, redirect=None):
		url = redirect if redirect != None else (self.base + start_url)
		return self._check_request(requests.get(url, headers=self.headers))

	def get_repos(self, redirect=None):
		return self._get_paginated(f"orgs/{self.org}/repos?type=all&per_page=100", redirect)

	def get_issues(self, repo, redirect=None):
		return self._get_paginated(f"repos/{self.org}/{repo}/issues?state=all&per_page=100", redirect)

	def get_next_page_url(self, link_header): # Link header format: <url>; rel=[prev|next|last], ...
		if (link_header == None):
			return None

		try:
			for (url, rel) in [x.split(';') for x in link_header.split(',')]:
				if (rel.strip().split('=')[1].strip('\"') == "next"): # Split 'rel=[prev|next|last]'
					return url.strip().replace('<', '').replace('>', '')
		except ValueError as e:
			pass

		return None

	def is_last_page(self, headers):
		return 'Link' not in headers or 'next' not in headers['Link']

	def repo_creation_to_date(self, s): # format : [Y]-[M]-[D]T[H]:[M]:[S]Z
		return datetime.strptime(s, '%Y-%m-%dT%H:%M:%SZ').date()

	def scrape_repos(self, org):
		self.org = org
		logging.info(f"Fetching all public repos from '{self.org}'...")
		repos = []
		req = requests.Response()
		req.headers = {'Link': 'next'} # Run loop at least once
		while not(self.is_last_page(req.headers)):
			next_page_url = self.get_next_page_url(req.headers['Link'])
			req = self.get_repos(next_page_url)
			repos += req.json()
			logging.debug(f"Got {len(repos)} repos, page {'1' if next_page_url == None else next_page_url[next_page_url.rindex('=')+1:]}")
		logging.info(f"Fetched {len(repos)} repos from '{self.org}' [success]")

		# Keep only audits reports starting from 20 March 2021 (earlier repos used a different format for tracking contributions)
		repos = list(filter(lambda repo: "-" in repo['name'] and self.repo_creation_to_date(repo['created_at']) > date(2021, 3, 20), repos))
		if (len(repos) == 0):
			logging.critical(f"No completed audits repos found, terminating...")
			exit(1)

		total_repos_size = sum([repo['size'] for repo in repos])
		logging.info(f"Found {len(repos)} completed audits repos (total size: {total_repos_size} Kb)")
		
		repos_data_folder = f"data/repos_data_{org}/"
		os.makedirs(repos_data_folder, exist_ok=True) # Create cloning directory if needed
		cloned_repos = 0
		logging.info(f"Cloning new repositories to '{repos_data_folder}'...")
		for repo in repos:
			if not(os.path.isdir(repos_data_folder + repo['name'])):
				logging.info(f"Cloning {repo['name']} ({repos.index(repo) + 1}/{len(repos)})...")
				Repo.clone_from(repo['clone_url'], repos_data_folder + repo['name'])
				cloned_repos += 1

		if (cloned_repos > 0):
			logging.info(f"Cloned {cloned_repos} new repos to '{repos_data_folder}' [success]")
		else:
			logging.warning(f"No new repos to clone")

		logging.info("Getting issues data for each repo (this may take some time)...")
		issues = {repo['name'] : [] for repo in repos}
		self.console_handler.terminator = "\r"
		for repo in repos:
			req = requests.Response()
			req.headers = {'Link': 'next'} # Run loop at least once
			count_repo_issues = 0
			while not(self.is_last_page(req.headers)):
				next_page_url = self.get_next_page_url(req.headers['Link'])
				req = self.get_issues(repo['name'], next_page_url)
				issues[repo['name']] += req.json()
				count_repo_issues += len(issues[repo['name']])
				logging.debug(f"Got {count_repo_issues} issues for repo '{repo['name']}', page {'1' if next_page_url == None else next_page_url[next_page_url.rindex('=')+1:]}")
			logging.info(f"Processed {repos.index(repo) + 1} / {len(repos)} repos")
		self.console_handler.terminator = "\n"
		logging.info(f"Got {sum([len(k) for k in issues.values()])} total issues in {len(repos)} repos from {self.org} [success]")

		'''
		At this point we have for each public contest report:
			- Sponsor
			- Rough date for when it took place (month, year)
			- Participants
				- Handle
				- Address
				- Issues reported
			- Issues (= audit submission) tags
				- Risk (QA, Non-critical/0, Low/1, Med/2, High/3)
				- Sponsor acknowledged, confirmed, disputed, addressed/resolved
				- Duplicate
				- Is gas optimization
				- Is judged invalid
				- Has been upgraded by judge
				- Has been withdrawn by warden
				... others
		'''

		logging.info(f"Parsing cloned repos data (this may take some time)...")
		repos_columns = ['contest', 'contest_sponsor', 'date', 'handle', 'address', 'risk', 'title', 'issueId', 'issueUrl', 'tags']
		repos_data = pd.DataFrame(columns=repos_columns)
		self.console_handler.terminator = "\n"

		return repos_data.reset_index(drop=True)



def scrape(scrape_method, scrape_data_desc, url, csv_file=None):
	logging.info(f"Starting {scrape_data_desc} data scraping at '{url}'...")
	df = scrape_method(url)
	
	logging.info(f"Finished {scrape_data_desc} data scraping: got {len(df.index)} rows of data [success]")
	return df	

if __name__ == "__main__":


	github_orgs = ["sherlock-audit", "code-423n4"]

	for g_org in github_orgs:
	

		file_handler = logging.FileHandler(f"logs/{g_org}.log", mode='w', encoding='utf8')
		console_handler = logging.StreamHandler()
		console_handler.setLevel(logging.INFO)
		logging.basicConfig(
			handlers=[file_handler, console_handler], 
			level=logging.DEBUG, 
			format='%(module)s:T+%(relativeCreated)d\t%(levelname)s %(message)s'
		)
		logging.getLogger('selenium').setLevel(logging.WARNING) # Prevent log file from being filed with Selenium debug output

		logging.addLevelName(logging.DEBUG, '[DEBUG]')
		logging.addLevelName(logging.INFO, '[*]')
		logging.addLevelName(logging.WARNING, '[!]')
		logging.addLevelName(logging.ERROR, '[ERROR]')
		logging.addLevelName(logging.CRITICAL, '[CRITICAL]')
		
		github_scraper = GithubScraper(console_handler, g_org)
		scrape(github_scraper.scrape_repos, f"Github {g_org} repos", g_org)