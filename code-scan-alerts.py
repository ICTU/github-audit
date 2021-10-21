import configparser

from github import Github
import github.GithubException

try:
    import github.CodeScanAlert
    using_github_package = True
except ImportError:
    from github.Requester import Requester
    from github.PaginatedList import PaginatedList
    from github_addition.CodeScanAlert import CodeScanAlert
    using_github_package = False


CONFIG_FILE = ".audit.cfg"

config = configparser.ConfigParser()
config.read(CONFIG_FILE)

org_name = config["github.com"].get("organization", "ICTU")


if using_github_package:
    def get_codescanning_alerts_for_repo(repo):
        return [alert for alert in repo.get_codescan_alerts()]
else:
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

    def get_codescanning_alerts_for_repo(repo):
        return [alert for alert in PaginatedList(CodeScanAlert, requester, f"{repo.url}/code-scanning/alerts", {})]


g = Github(config["github.com"]["token"])

org = g.get_organization(org_name)

print(f"Using {'only the github package' if using_github_package else 'the local github package additions'}")

repo_alerts = {}
repo_alerts_exceptions = {}
for repo in org.get_repos(sort="full_name"):
    try:
        repo_alerts[repo.full_name] = get_codescanning_alerts_for_repo(repo)
    except github.GithubException as e:
        repo_alerts_exceptions.setdefault(e.data['message'], []).append(repo.full_name)

if not repo_alerts_exceptions:
    print("no repos without code scanning alerts")
else:
    for msg in sorted(repo_alerts_exceptions):
        print(msg)
        for repo in sorted(repo_alerts_exceptions[msg]):
            print(f"  - {repo}")

print()

if not repo_alerts:
    print("no repos with code scanning alerts")
else:
    print("T(ool) - R(ule) - A(lert) - H(istory of alerts) for repositories:")
    for repo, alerts in sorted(repo_alerts.items()):
        print(repo)
        for alert in sorted(alerts, key=lambda alert: alert.number, reverse=True):
            print(f"  # {alert.number}"
                  f" : {alert.created_at if alert.created_at else '--'}"
                  f" | {alert.dismissed_at if alert.dismissed_at else '--'}")
            print(f"    T {alert.tool.name} {alert.tool.version} | {alert.tool.guid}")
            print(f"    R {alert.rule.name} | {alert.rule.security_severity_level} | {alert.rule.severity}")
            print(f"      {alert.rule.description}")
            print(f"    A {alert.most_recent_instance.ref} | {alert.most_recent_instance.state} | {alert.most_recent_instance.commit_sha}")
            print(f"      {alert.most_recent_instance.location}")
            print(f"      {alert.most_recent_instance.analysis_key}")
            print(f"      {alert.most_recent_instance.message.get('text', '--')}")
            # only show instances when the state changes
            last_state = None
            for nr, instance in enumerate(alert.get_instances(), 1):
                if last_state == instance.state:
                    continue
                print(f"    H {nr:04d} {instance.ref} | {instance.state} | {instance.commit_sha}")
                print(f"          {instance.location}")
                print(f"          {instance.analysis_key}")
                print(f"          {instance.message.get('text', '--')}")
                last_state = instance.state
