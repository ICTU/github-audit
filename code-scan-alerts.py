import configparser

import pprint

import github.GithubException
from github import Github
from github.Requester import Requester
from github.PaginatedList import PaginatedList

from GithubCodeScanAlert import CodeScanAlert


CONFIG_FILE = ".audit.cfg"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

org_name = config["github.com"].get("organization", "ICTU")


def class_creation_arguments(requester, headers, element, *args, **kwargs):
    return {
        "requester": requester,
        "headers": headers,
        "element": element,
        "args": args,
        "kwargs": kwargs,
    }


g = Github(config["github.com"]["token"])

org = g.get_organization(org_name)

requester = Requester(
    login_or_token=config["github.com"]["token"],
    password=None,
    jwt=None,
    base_url="https://api.github.com",
    timeout=15,
    user_agent="PyGithub/Python",
    per_page=30,
    verify=True,
    retry=None,
    pool_size=None,
)

repo_alerts = {}
repo_alerts_exceptions = {}
for repo in org.get_repos(sort="full_name"):
    try:
        repo_alerts[repo.full_name] = [
            alert
            for alert in PaginatedList(
                CodeScanAlert,
                requester,
                f"https://api.github.com/repos/{org_name}/{repo.name}/code-scanning/alerts",
                {},
            )
        ]
    except github.GithubException as e:
        repo_alerts_exceptions.setdefault(e.data['message'], []).append(repo.full_name)

for msg in sorted(repo_alerts_exceptions):
    print(msg)
    for repo in sorted(repo_alerts_exceptions[msg]):
        print(f"  - {repo}")

if repo_alerts_exceptions and repo_alerts:
    print()

if repo_alerts:
    print("T(ool) - R(ule) - F(inding) for repositories:")
for repo, alerts in sorted(repo_alerts.items()):
    print(repo)
    for alert in sorted(alerts, key=lambda alert: alert.number, reverse=True):
        print(f"  # {alert.number}"
              f" : {alert.created_at if alert.created_at else '--'}"
              f" | {alert.dismissed_at if alert.dismissed_at else '--'}")
        print(f"    T {alert.tool.get('name', '--')}")
        print(f"    R {alert.rule.get('security_severity_level', '--')} | {alert.rule.get('severity', '--')}")
        print(f"      {alert.rule.get('description', '--')}")
        print(f"    F {alert.most_recent_instance.get('ref', '--')} | {alert.most_recent_instance.get('state', '--')}")
        print(f"      {alert.most_recent_instance.get('message', dict()).get('text', '--')}")
