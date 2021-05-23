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
from yattag import Doc, indent


CONFIG_FILE = ".audit.cfg"
# supported:
# [github.com]
# organization=NAME -- optional
# token=TOKEN -- required


class OutputFormat(str, Enum):
    text = "text"
    json = "json"
    html = "html"

    @staticmethod
    def unknown(what):
        print(f"PANIC! unknown output format {what}")


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
    """Convert the boolean to a string."""
    return "\N{BALLOT BOX WITH CHECK}" if boolean else ""


def format_int(integer: Optional[int]) -> str:
    """Convert the integer to a string."""
    return "" if integer is None else str(integer)


def format_timestamp(timestamp: str, now: datetime) -> str:
    return f'{timestamp} ({timeago.format(datetime.fromisoformat(timestamp), now)})'


def format_member(member) -> str:
    """Return the member name and/or login."""
    name = member.get("name") or ""
    login = member.get("login") or ""
    return f"{name} ({login})" if name and login else name or login


def format_email_href(email):
    return f'<a href="mailto:{email}">{email}</a>' if email else ""


def format_url_href(url):
    return f'<a href="{url}">{url}</a>' if url else ""


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
    elif output_format == OutputFormat.html:
        now = datetime.now()
        title = f"Github repositories of {organization_name}"
        doc, tag, text = Doc().tagtext()
        with tag("html"):
            with tag("head"):
                doc.asis('<meta charset="UTF-8">')
                with tag("title"):
                    text(title)
                with tag("style"):
                    text("\n.centered {text-align: center}")
                    text("\ntable {border-spacing: 0; border-collapse: collapse; }")
                    text("\n.first-row {border-top: 1px solid black; }")
            with tag("body"):
                with tag("h1"):
                    text(title)
                with tag("table"):
                    with tag("tr"):
                        with tag("th", rowspan=2):
                            text("Name")
                        empty_columns = 2  # Name, Pushed at
                        if include_archived_repositories:
                            with tag("th", rowspan=2):
                                text("Archived")
                            empty_columns += 1
                        if include_forked_repositories:
                            with tag("th", rowspan=2):
                                text("Fork")
                            empty_columns += 1
                        with tag("th", rowspan=2):
                            text("Pushed at")
                        with tag("th", colspan=2):
                            doc.attr(style="border-bottom: 1px solid black;")
                            text("Pull request")
                    with tag("tr"):
                        with tag("th"):
                            text("Title")
                        with tag("th"):
                            text("Created at")
                    for repo in sorted(repositories, key=lambda x: x['name'].lower()):
                        with tag("tr"):
                            with tag("td", klass="first-row"):
                                text(repo["name"])
                            if include_archived_repositories:
                                with tag("td", klass="first-row centered"):
                                    doc.asis(format_bool(repo["archived"]))
                            if include_forked_repositories:
                                with tag("td", klass="first-row centered"):
                                    doc.asis(format_bool(repo["fork"]))
                            with tag("td", klass="first-row"):
                                text(format_timestamp(repo["pushed_at"], now))
                            if repo["open_pull_requests"]:
                                first_pr = repo["open_pull_requests"][0]
                                with tag("td", klass="first-row"):
                                    text(first_pr["title"])
                                with tag("td", klass="first-row"):
                                    text(format_timestamp(first_pr["created_at"], now))
                            else:
                                doc.stag("td", klass="first-row", colspan=2)
                        for pr in repo["open_pull_requests"][1:]:
                            with tag("tr"):
                                doc.stag("td", colspan=empty_columns)
                                with tag("td"):
                                    text(pr["title"])
                                with tag("td"):
                                    text(format_timestamp(pr["created_at"], now))
        print(indent(doc.getvalue()))
    elif output_format == OutputFormat.text:
        now = datetime.now()
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
        for repo in sorted(repositories, key=lambda x: x['name'].lower()):
            repo_row = [repo["name"]]
            if include_archived_repositories:
                repo_row.append(format_bool(repo["archived"]))
            if include_forked_repositories:
                repo_row.append(format_bool(repo["fork"]))
            repo_row.append(f'{format_timestamp(repo["pushed_at"], now)}')
            if repo["open_pull_requests"]:
                first_pr = repo["open_pull_requests"][0]
                repo_row.extend([first_pr["title"], f'{format_timestamp(first_pr["created_at"], now)}'])
            table.add_row(*repo_row)
            for pr in repo["open_pull_requests"][1:]:
                pr_row = empty_columns + [pr["title"], f'{format_timestamp(pr["created_at"], now)}']
                table.add_row(*pr_row)
        Console().print(table)
    else:
        OutputFormat.unknown(output_format)


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
                dict(
                    contributions=contributor.contributions,
                    login=contributor.login, name=contributor.name,
                    email=contributor.email, url=contributor.html_url,
                )
                for contributor in get_contributors(repo)
            ]
        )
        for repo in get_repos(organization, include_forked_repositories, include_archived_repositories)
    ]
    if output_format == OutputFormat.json:
        repositories.sort(key=lambda repo: repo["name"] or "")
        echo_json(repositories)
    elif output_format == OutputFormat.html:
        title = f"Contributors to github repositories of {organization_name}"
        doc, tag, text = Doc().tagtext()
        with tag('html'):
            with tag("head"):
                doc.asis('<meta charset="UTF-8">')
                with tag("title"):
                    text(title)
                with tag("style"):
                    text("\n.right {text-align: right; }")
                    text("\ntable {border-spacing: 0; border-collapse: collapse; }")
                    text("\n.first-row {border-top: 1px solid black; }")
            with tag("body"):
                with tag("h1"):
                    text(title)
                with tag("table"):
                    with tag("tr"):
                        with tag("th", rowspan=2):
                            text("Name")
                        with tag("th", colspan=4):
                            doc.attr(style="border-bottom: 1px solid black;")
                            text("Contributor")
                        with tag("th", rowspan=2):
                            doc.asis("Nr. of<br/>contributions")
                    with tag("tr"):
                        with tag("th"):
                            text("Name")
                        with tag("th"):
                            text("Login")
                        with tag("th"):
                            text("Email")
                        with tag("th"):
                            text("Profile")
                    for repo in sorted(repositories, key=lambda x: x['name'].lower()):
                        contributors = sorted(
                            repo['contributors'],
                            key=lambda x: (1_000_000_000 - x['contributions'], x['login'].lower())
                        )
                        first_contributor = contributors[0] if contributors else dict()
                        name = first_contributor.get("name") or ""
                        login = first_contributor.get("login") or ""
                        email = first_contributor.get("email") or ""
                        url = first_contributor.get("url") or ""
                        with tag("tr"):
                            with tag("td", klass="first-row"):
                                text(repo["name"])
                            with tag("td", klass="first-row"):
                                text(name)
                            with tag("td", klass="first-row"):
                                text(login)
                            with tag("td", klass="first-row"):
                                doc.asis(format_email_href(email))
                            with tag("td", klass="first-row"):
                                doc.asis(format_url_href(url))
                            with tag("td", klass="first-row right"):
                                text(format_int(first_contributor.get("contributions")))
                        for contributor in contributors[1:]:
                            name = contributor.get("name") or ""
                            login = contributor.get("login") or ""
                            email = contributor.get("email") or ""
                            url = contributor.get("url") or ""
                            with tag("tr"):
                                doc.stag("td")
                                with tag("td"):
                                    text(name)
                                with tag("td"):
                                    text(login)
                                with tag("td"):
                                    doc.asis(format_email_href(email))
                                with tag("td"):
                                    doc.asis(format_url_href(url))
                                with tag("td", klass="right"):
                                    text(format_int(contributor["contributions"]))
        print(indent(doc.getvalue()))
    elif output_format == OutputFormat.text:
        table = Table(
            "Name", "Contributor", Column("Nr. of contributions", justify="right"), box=box.SQUARE
        )
        for repo in sorted(repositories, key=lambda x: x['name'].lower()):
            contributors = sorted(repo['contributors'], key=lambda x: (1_000_000_000 - x['contributions'], x['login'].lower()))
            first_contributor = contributors[0] if contributors else dict()
            first_row = [repo["name"], format_member(first_contributor), format_int(first_contributor.get("contributions"))]
            table.add_row(*first_row)
            for contributor in contributors[1:]:
                table.add_row(None, format_member(contributor), format_int(contributor["contributions"]))
        Console().print(table)
    else:
        OutputFormat.unknown(output_format)


@app.command()
def members(
        organization_name: str = organization_name_argument,
        output_format: OutputFormat = output_format_option
):
    organization = g.get_organization(organization_name)
    member_info = [
        dict(
            login=member.login, name=member.name,
            email=member.email, url=member.html_url,
            membership_state=membership.state, membership_role=membership.role,
        )
        for member, membership in get_members_and_membership(organization)
    ]
    if output_format == OutputFormat.json:
        member_info.sort(key=lambda member: member["name"] or "")
        echo_json(member_info)
    elif output_format == OutputFormat.html:
        title = f"Members of {organization_name} on github"
        doc, tag, text = Doc().tagtext()
        with tag('html'):
            with tag("head"):
                doc.asis('<meta charset="UTF-8">')
                with tag("title"):
                    text(title)
                with tag("style"):
                    text("\n.centered {text-align: center}")
                    text("\ntable {border-spacing: 0; border-collapse: collapse; }")
            with tag("body"):
                with tag("h1"):
                    text(title)
                with tag("table"):
                    with tag("tr"):
                        with tag("th", colspan=4):
                            doc.attr(style="border-bottom: 1px solid black;")
                            text("Member")
                        doc.stag("th")
                        with tag("th", colspan=2):
                            doc.attr(style="border-bottom: 1px solid black;")
                            text("Membership")
                    with tag("tr"):
                        with tag("th"):
                            text("Name")
                        with tag("th"):
                            text("Login")
                        with tag("th"):
                            text("Email")
                        with tag("th"):
                            text("Profile")
                        doc.stag("th")
                        with tag("th"):
                            text("state")
                        with tag("th"):
                            text("role")
                    for member in sorted(member_info, key=lambda x: x['login']):
                        name = member.get("name") or ""
                        login = member.get("login") or ""
                        email = member.get("email") or ""
                        url = member.get("url") or ""
                        with tag("tr"):
                            with tag("td"):
                                text(name)
                            with tag("td"):
                                text(login)
                            with tag("td"):
                                doc.asis(format_email_href(email))
                            with tag("td"):
                                doc.asis(format_url_href(url))
                            doc.stag("td")
                            with tag("td"):
                                text(member['membership_state'])
                            with tag("td"):
                                text(member['membership_role'])
        print(indent(doc.getvalue()))
    elif output_format == OutputFormat.text:
        table = Table("Member", "Membership state", "Membership role", box=box.SQUARE)
        for member in sorted(member_info, key=lambda x: x['login']):
            table.add_row(format_member(member), member['membership_state'], member['membership_role'])
        Console().print(table)
    else:
        OutputFormat.unknown(output_format)


if __name__ == "__main__":
    app()
