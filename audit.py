"""Audit GitHub organizations."""

import configparser
import json
from typing import Optional
from enum import Enum

from github import Github
import typer


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


config = configparser.ConfigParser()
config.read(".audit.cfg")
g = Github(config["github.com"]["token"])
app = typer.Typer()


def make_repo_filter(include_forked_repositories: bool, include_archived_repositories: bool):
    def _filter(repo):
        return (repo.archived and not include_archived_repositories) or (repo.fork and not include_forked_repositories)
    return _filter


def filter_repos(repos, include_forked_repositories: bool, include_archived_repositories: bool):
    ignore_repo = make_repo_filter(include_forked_repositories, include_archived_repositories)
    for repo in repos:
        if not ignore_repo(repo):
            yield repo


@app.command()
def repos(
    organization_name: str, 
    include_forked_repositories: bool = typer.Option(True, "--include-forked-repositories/--exclude-forked-repositories", "-f/-F"), 
    include_archived_repositories: bool = typer.Option(False, "--include-archived-repositories/--exclude-archived-repositories", "-a/-A"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
):
    organization = g.get_organization(organization_name)
    repositories = []
    with typer.progressbar(organization.get_repos(), length=organization.public_repos) as repos:
        for repo in filter_repos(repos, include_forked_repositories, include_archived_repositories):
            open_prs = [
                dict(title=pr.title, created_at=pr.created_at.isoformat(),)
                for pr in repo.get_pulls()
            ]
            open_prs.sort(key=lambda pr: pr["created_at"])
            repositories.append(
                dict(
                    name=repo.name, full_name=repo.full_name, url=repo.html_url,
                    archived=repo.archived, fork=repo.fork, pushed_at=repo.pushed_at.isoformat(),
                    open_pull_requests=open_prs,
                )
            )
    if output_format == OutputFormat.json:
        repositories.sort(key=lambda repo: repo["pushed_at"])
        typer.echo(json.dumps(repositories, indent="  "))
    else:
        for repo in sorted(repositories, key=lambda x: x['name'].lower()):
            print(f"{repo['name']}\t{repo['archived']}\t{repo['fork']}\t{repo['pushed_at']}")
            for pr in repo['open_pull_requests']:
                print(f"\t{pr['title']}\t{pr['created_at']}")
            print()


@app.command()
def repo_contributions(
    organization_name: str, 
    include_forked_repositories: bool = typer.Option(True, "--include-forked-repositories/--exclude-forked-repositories", "-f/-F"), 
    include_archived_repositories: bool = typer.Option(False, "--include-archived-repositories/--exclude-archived-repositories", "-a/-A"),
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
):
    organization = g.get_organization(organization_name)
    repositories = []
    with typer.progressbar(organization.get_repos(), length=organization.public_repos) as repos:
        for repo in filter_repos(repos, include_forked_repositories, include_archived_repositories):
            repo_contributors = []
            for contributor in repo.get_contributors():
                if contributor.type.lower() == "bot":
                    continue
                repo_contributors.append(
                    dict(contributions=contributor.contributions, login=contributor.login, name=contributor.name,)
                )
            repositories.append(
                dict(
                    name=repo.name, full_name=rep.full_name, url=repo.html_url,
                    archived=repo.archived, fork=repo.fork, pushed_at=repo.pushed_at.isoformat(),
                    contributors=repo_contributors,
                )
            )
    if output_format == OutputFormat.json:
        repositories.sort(key=lambda repo: repo["name"] or "")
        typer.echo(json.dumps(repositories, indent="  "))
    else:
        for repo in sorted(repositories, key=lambda x: x['name'].lower()):
            print(f"{repo['name']}\t{repo['url']}")
            for contributor in sorted(repo['contributors'], key=lambda x: (1_000_000_000 - x['contributions'], x['login'].lower())):
                items = [ "", str(contributor['contributions']), contributor['login']]
                if contributor['name']:
                    items.append(contributor['name'])
                print("\t".join(items))
            print()


@app.command()
def members(
    organization_name: str,
    output_format: OutputFormat = typer.Option(OutputFormat.text, "--format"),
):
    organization = g.get_organization(organization_name)
    member_info = []
    with typer.progressbar(organization.get_members()) as members:
        for member in members:
            try:
                membership = member.get_organization_membership(organization)
            except:
                membership = None
            member_info.append(
                dict(login=member.login, name=member.name, membership_state=membership.state, membership_role=membership.role,)
            )
    if output_format == OutputFormat.json:
        member_info.sort(key=lambda member: member["name"] or "")
        typer.echo(json.dumps(member_info, indent="  "))
    else:
        for member in sorted(member_info, key=lambda x: x['login']):
            print(f"{member['login']}\t{member['name']}\t{member['membership_state']}\t{member['membership_role']}")


if __name__ == "__main__":
   app()

