"""Audit GitHub organizations."""

import configparser
import json
from enum import Enum
from typing import Optional
from datetime import datetime

from github import Github
from rich.console import Console
from rich.table import Table, Column
from rich import box
import typer
import timeago


CONFIG_FILE = ".audit.cfg"
# supported:
# [github.com]
# organization=NAME -- optional
# token=TOKEN -- required


class OutputFormat(str, Enum):
    text = "text"
    json = "json"


config = configparser.ConfigParser()
config.read(CONFIG_FILE)

organization_name_argument = typer.Argument(config["github.com"].get("organization", None))
output_format_option = typer.Option(OutputFormat.text, "--format")
fork_option = typer.Option(True, "--include-forked-repositories/--exclude-forked-repositories", "-f/-F")
archive_option = typer.Option(False, "--include-archived-repositories/--exclude-archived-repositories", "-a/-A")


g = Github(config["github.com"]["token"])
app = typer.Typer()


def get_repos(organization, include_forked_repositories: bool, include_archived_repositories: bool):
    """Return the repositories of the organization, displaying a scrollbar."""
    with typer.progressbar(organization.get_repos(), length=organization.public_repos) as repos:
        for repo in repos:
            if (repo.archived and not include_archived_repositories) or (repo.fork and not include_forked_repositories):
                continue
            yield repo


def get_contributors(repo):
    """Return the non-bot contributors to a repository"""
    for contributor in repo.get_contributors():
        if contributor.type.lower() != "bot":
            yield contributor


def get_members_and_membership(organization):
    """Return the members of the organization."""
    with typer.progressbar(organization.get_members()) as members:
        for member in members:
            yield member, member.get_organization_membership(organization)


def echo_json(json_data) -> None:
    """Output the json data to stdout."""
    typer.echo(json.dumps(json_data, indent="  "))


def format_bool(boolean: bool) -> str:
    """Format the boolean."""
    return "\N{BALLOT BOX WITH CHECK}" if boolean else ""


def format_int(integer: Optional[int]) -> str:
    """Format the integer."""
    return "" if integer is None else str(integer)


def human_friendly_timestamp(timestamp: str, now: datetime):
    return f'{timestamp} ({timeago.format(datetime.fromisoformat(timestamp), now)})'


@app.command()
def repos(
    organization_name: str = organization_name_argument,
    include_forked_repositories: bool = fork_option,
    include_archived_repositories: bool = archive_option,
    output_format: OutputFormat = output_format_option,
):
    organization = g.get_organization(organization_name)
    repositories = []
    for repo in get_repos(organization, include_forked_repositories, include_archived_repositories):
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
        echo_json(repositories)
    else:
        table = Table("Name", box=box.SQUARE)
        empty_columns = [None, None]  # Name, Pushed at
        if include_archived_repositories:
            table.add_column("Archived", justify="center")
            empty_columns.append(None)
        if include_forked_repositories:
            table.add_column("Fork", justify="center")
            empty_columns.append(None)
        table.add_column("Pushed at")
        table.add_column("Pull request title")
        table.add_column("Created at")
        now = datetime.now()
        for repo in sorted(repositories, key=lambda x: x['name'].lower()):
            repo_row = [repo["name"]]
            if include_archived_repositories:
                repo_row.append(format_bool(repo["archived"]))
            if include_forked_repositories:
                repo_row.append(format_bool(repo["fork"]))
            repo_row.append(f'{human_friendly_timestamp(repo["pushed_at"], now)}')
            if repo["open_pull_requests"]:
                first_pr = repo["open_pull_requests"][0]
                repo_row.extend([first_pr["title"], f'{human_friendly_timestamp(first_pr["created_at"], now)}'])
            table.add_row(*repo_row)
            for pr in repo["open_pull_requests"][1:]:
                pr_row = empty_columns + [pr["title"], f'{human_friendly_timestamp(pr["created_at"], now)}']
                table.add_row(*pr_row)
        Console().print(table)


@app.command()
def repo_contributions(
    organization_name: str = organization_name_argument,
    include_forked_repositories: bool = fork_option,
    include_archived_repositories: bool = archive_option,
    output_format: OutputFormat = output_format_option,
):
    organization = g.get_organization(organization_name)
    repositories = [
        dict(
            name=repo.name, full_name=repo.full_name, url=repo.html_url,
            archived=repo.archived, fork=repo.fork, pushed_at=repo.pushed_at.isoformat(),
            contributors=[
                dict(contributions=contributor.contributions, login=contributor.login, name=contributor.name, )
                for contributor in get_contributors(repo)
            ]
        )
        for repo in get_repos(organization, include_forked_repositories, include_archived_repositories)
    ]
    if output_format == OutputFormat.json:
        repositories.sort(key=lambda repo: repo["name"] or "")
        echo_json(repositories)
    else:
        table = Table(
            "Name", "Contributor name", "Contributor login",
            Column("Nr. of contributions", justify="right"),
            box=box.SQUARE
        )
        for repo in sorted(repositories, key=lambda x: x['name'].lower()):
            contributors = sorted(repo['contributors'], key=lambda x: (1_000_000_000 - x['contributions'], x['login'].lower()))
            first_contributor = contributors[0] if contributors else dict()
            first_row = [repo["name"], first_contributor.get("name"), first_contributor.get("login"), format_int(first_contributor.get("contributions"))]
            table.add_row(*first_row)
            for contributor in contributors[1:]:
                table.add_row(None, contributor.get("name"), contributor["login"], format_int(contributor["contributions"]))
        Console().print(table)


@app.command()
def members(
        organization_name: str = organization_name_argument,
        output_format: OutputFormat = output_format_option
):
    organization = g.get_organization(organization_name)
    member_info = [
        dict(login=member.login, name=member.name, membership_state=membership.state, membership_role=membership.role, )
        for member, membership in get_members_and_membership(organization)
    ]
    if output_format == OutputFormat.json:
        member_info.sort(key=lambda member: member["name"] or "")
        echo_json(member_info)
    else:
        table = Table("Login", "Name", "Membership state", "Membership role", box=box.SQUARE)
        for member in sorted(member_info, key=lambda x: x['login']):
            table.add_row(member['login'], member['name'], member['membership_state'], member['membership_role'])
        Console().print(table)


if __name__ == "__main__":
    app()
