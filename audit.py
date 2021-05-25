"""Audit GitHub organizations."""

import configparser
import json
from enum import Enum
from typing import Optional, Tuple
from datetime import datetime

from github import Github, Organization, Repository, NamedUser, Membership
from rich.console import Console
from rich.table import Table, Column
from rich import box
import typer
import timeago
from yattag import SimpleDoc, Doc, indent


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
    def unknown(what: str) -> None:
        print(f"PANIC! unknown output format {what}")


config = configparser.ConfigParser()
config.read(CONFIG_FILE)

organization_name_argument = typer.Argument(config["github.com"].get("organization", None))
output_format_option = typer.Option(OutputFormat.text, "--format")
fork_option = typer.Option(True, "--include-forked-repositories/--exclude-forked-repositories", "-f/-F")
archive_option = typer.Option(False, "--include-archived-repositories/--exclude-archived-repositories", "-a/-A")
output_option = typer.Option(None, "--output", "-o")


g = Github(config["github.com"]["token"])
app = typer.Typer()


# Github API wrappers

def get_repos(
        organization: Organization,
        include_forked_repositories: bool,
        include_archived_repositories: bool
) -> Repository:
    """Return the repositories of the organization, displaying a scrollbar."""
    with typer.progressbar(organization.get_repos(), length=organization.public_repos, label="Collecting") as repos:
        for repo in repos:
            if repo.archived and not include_archived_repositories:
                continue
            if repo.fork and not include_forked_repositories:
                continue
            yield repo


def get_contributors(repo: Repository) -> NamedUser:
    """Return the non-bot contributors to a repository"""
    for contributor in repo.get_contributors():
        if contributor.type.lower() != "bot":
            yield contributor


def get_members_and_membership(organization) -> Tuple[NamedUser, Membership]:
    """Return the members of the organization and their membership."""
    with typer.progressbar(organization.get_members(), label="Collecting") as members:
        for member in members:
            yield member, member.get_organization_membership(organization)


# formatting utilities

def format_bool(boolean: bool) -> str:
    """Convert a boolean to a string."""
    return "\N{BALLOT BOX WITH CHECK}" if boolean else ""


def format_int(integer: Optional[int]) -> str:
    """Convert the integer to a string."""
    return "" if integer is None else str(integer)


def format_friendly_timestamp(timestamp: str, now: datetime) -> str:
    """Add a user friendly relative time to a timestamp."""
    return f'{timestamp} ({timeago.format(datetime.fromisoformat(timestamp), now)})'


def format_generated_timestamp(dt: datetime) -> str:
    """Return standard phrase for the date and time the report is generated"""
    dt_as_text = dt.astimezone().strftime('%Y/%m/%d %H:%M:%S %Z')
    return f"generated on {dt_as_text}"


def format_member(member: NamedUser) -> str:
    """Return the member name and/or login."""
    name = member.get("name") or ""
    login = member.get("login") or ""
    return f"{name} ({login})" if name and login else name or login


# HTML utilities

def create_html_head(doc: SimpleDoc, title: str) -> None:
    """
    Standard HTML head segment
    - sets text encoding for compatibility with Python
    - sets document title
    - defines styles for content
    :param doc: yattag document
    :param title: document title
    """
    with doc.tag("head"):
        doc.asis('<meta charset="UTF-8">')
        with doc.tag("title"):
            doc.text(title)
        with doc.tag("style"):
            doc.text("\nbody {font-size: 100%; }")
            doc.text("\ndiv.footer {font-size: 50%; padding-top: 24px; }")
            doc.text("\ntable {border-spacing: 0; border-collapse: collapse; }")
            doc.text("\nth {vertical-align: bottom; }")
            doc.text("\ntd {vertical-align: top; }")
            doc.text("\n.centered {text-align: center; }")
            doc.text("\n.column-group-header {border-bottom: 1px solid black; }")
            doc.text("\n.first-row {border-top: 1px solid black; }")
            doc.text("\n.right {text-align: right; }")


def create_html_email_href(email: str) -> str:
    """
    HTML version of an email address
    :param email: the email address
    :return: email address for use in an HTML document
    """
    return f'<a href="mailto:{email}">{email}</a>' if email else ""


def create_html_url_href(url: str) -> str:
    """
    HTML version of a URL
    :param url: the URL
    :return: URL for use in an HTML document
    """
    return f'<a href="{url}">{url}</a>' if url else ""


def create_html_footer(doc: SimpleDoc, started: datetime) -> None:
    """
    Standard HTML footer for the body segment
    :param doc: yattag document
    :param started: when report creation started
    """
    with doc.tag("div", klass="footer"):
        doc.text(format_generated_timestamp(started))


# output creation

def output_json(json_data: list, output: Optional[typer.FileTextWrite]) -> None:
    """
    Output the json data
    :param json_data: data to convert to JSON
    :param output: output file to write to (default: stdout)
    """
    typer.echo(json.dumps(json_data, indent="  "), file=output)


def output_text_elements(title: str, table: Table, started: datetime, output: Optional[typer.FileTextWrite]) -> None:
    """
    Output the text elements
    :param title: title above the table
    :param table: data to report
    :param started: when creating this report started
    :param output: output file to write to (default: stdout)
    """
    console = Console(file=output)
    console.print(title)
    console.print()
    console.print(table)
    console.print()
    console.print(format_generated_timestamp(started))


@app.command()
def repos(
        organization_name: str = organization_name_argument,
        include_forked_repositories: bool = fork_option,
        include_archived_repositories: bool = archive_option,
        output_format: OutputFormat = output_format_option,
        output: typer.FileTextWrite = output_option,
) -> None:
    started = datetime.now()
    title = f"Github repositories of {organization_name}"

    organization = g.get_organization(organization_name)
    repositories = [
        dict(
            name=repo.name, full_name=repo.full_name, url=repo.html_url,
            archived=repo.archived, fork=repo.fork, pushed_at=repo.pushed_at.isoformat(),
            open_pull_requests=[
                dict(
                    title=pr.title, created_at=pr.created_at.isoformat(),
                )
                for pr in sorted(repo.get_pulls(), reverse=True, key=lambda pr: pr.created_at)
            ],
        )
        for repo in sorted(
            get_repos(organization, include_forked_repositories, include_archived_repositories),
            reverse=True,
            key=lambda repo: repo.pushed_at
        )
    ]

    if output_format == OutputFormat.json:
        output_json(repositories, output)

    elif output_format == OutputFormat.html:
        doc, tag, text = Doc().tagtext()
        with tag("html"):
            with tag("head"):
                create_html_head(doc, title)
            with tag("body"):
                with tag("h1"):
                    text(title)
                with tag("table"):
                    with tag("tr"):
                        with tag("th", rowspan=2):
                            text("Name")
                        if include_archived_repositories:
                            with tag("th", rowspan=2):
                                text("Archived")
                        if include_forked_repositories:
                            with tag("th", rowspan=2):
                                text("Fork")
                        with tag("th", rowspan=2):
                            text("Pushed at")
                        with tag("th", colspan=2, klass="column-group-header"):
                            text("Pull request")
                    with tag("tr"):
                        with tag("th"):
                            text("Title")
                        with tag("th"):
                            text("Created at")
                    for repo in sorted(repositories, key=lambda x: x['name'].lower()):
                        open_pull_requests = repo["open_pull_requests"]
                        with tag("tr"):
                            with tag("td", klass="first-row", rowspan=len(open_pull_requests)):
                                text(repo["name"])
                            if include_archived_repositories:
                                with tag("td", klass="first-row centered", rowspan=len(open_pull_requests)):
                                    doc.asis(format_bool(repo["archived"]))
                            if include_forked_repositories:
                                with tag("td", klass="first-row centered", rowspan=len(open_pull_requests)):
                                    doc.asis(format_bool(repo["fork"]))
                            with tag("td", klass="first-row", rowspan=len(open_pull_requests)):
                                text(format_friendly_timestamp(repo["pushed_at"], started))
                            if open_pull_requests:
                                first_pr = open_pull_requests[0]
                                with tag("td", klass="first-row"):
                                    text(first_pr["title"])
                                with tag("td", klass="first-row"):
                                    text(format_friendly_timestamp(first_pr["created_at"], started))
                            else:
                                doc.stag("td", klass="first-row", colspan=2)
                        for pr in open_pull_requests[1:]:
                            with tag("tr"):
                                with tag("td"):
                                    text(pr["title"])
                                with tag("td"):
                                    text(format_friendly_timestamp(pr["created_at"], started))
                create_html_footer(doc, started)
        print(indent(doc.getvalue()), file=output)

    elif output_format == OutputFormat.text:
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
            repo_row.append(f'{format_friendly_timestamp(repo["pushed_at"], started)}')
            if repo["open_pull_requests"]:
                first_pr = repo["open_pull_requests"][0]
                repo_row.extend([first_pr["title"], f'{format_friendly_timestamp(first_pr["created_at"], started)}'])
            table.add_row(*repo_row)
            for pr in repo["open_pull_requests"][1:]:
                pr_row = empty_columns + [pr["title"], f'{format_friendly_timestamp(pr["created_at"], started)}']
                table.add_row(*pr_row)
        output_text_elements(title, table, started, output)

    else:
        OutputFormat.unknown(output_format)


@app.command()
def repo_contributions(
        organization_name: str = organization_name_argument,
        include_forked_repositories: bool = fork_option,
        include_archived_repositories: bool = archive_option,
        output_format: OutputFormat = output_format_option,
        output: typer.FileTextWrite = output_option,
) -> None:
    started = datetime.now()
    title = f"Contributions to github repositories of {organization_name}"

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
                for contributor in sorted(
                    get_contributors(repo),
                    key=lambda contributor: (1_000_000_000 - contributor.contributions, contributor.login.lower())
                )
            ]
        )
        for repo in sorted(
            get_repos(organization, include_forked_repositories, include_archived_repositories),
            key=lambda repo: repo.name or ""
        )
    ]

    if output_format == OutputFormat.json:
        output_json(repositories, output)

    elif output_format == OutputFormat.html:
        doc, tag, text = Doc().tagtext()
        with tag('html'):
            create_html_head(doc, title)
            with tag("body"):
                with tag("h1"):
                    text(title)
                with tag("table"):
                    with tag("tr"):
                        with tag("th", rowspan=2):
                            text("Name")
                        with tag("th", colspan=5, klass="column-group-header"):
                            text("Contributor")
                    with tag("tr"):
                        with tag("th"):
                            text("Name")
                        with tag("th"):
                            text("Login")
                        with tag("th"):
                            text("Email")
                        with tag("th"):
                            text("Profile")
                        with tag("th"):
                            text("#contributions")

                    def contributor_row_cells(contributor, klass):
                        td_optional_klass_arg = dict(klass=klass) if klass else {}
                        name = contributor.get("name") or ""
                        login = contributor.get("login") or ""
                        email = contributor.get("email") or ""
                        url = contributor.get("url") or ""
                        with tag("td", **td_optional_klass_arg):
                            text(name)
                        with tag("td", **td_optional_klass_arg):
                            text(login)
                        with tag("td", **td_optional_klass_arg):
                            doc.asis(create_html_email_href(email))
                        with tag("td", **td_optional_klass_arg):
                            doc.asis(create_html_url_href(url))
                        with tag("td", klass=f"{klass} right" if klass else "right"):
                            text(format_int(contributor.get("contributions")))

                    for repo in repositories:
                        contributors = repo['contributors']
                        if len(contributors) == 0:
                            contributors = [{}]
                        with tag("tr"):
                            with tag("td", klass="first-row", rowspan=len(contributors)):
                                text(repo["name"])
                            contributor_row_cells(contributors[0], "first-row")
                        for contributor in contributors[1:]:
                            with tag("tr"):
                                contributor_row_cells(contributor, None)
                create_html_footer(doc, started)
        print(indent(doc.getvalue()), file=output)

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
        output_text_elements(title, table, started, output)

    else:
        OutputFormat.unknown(output_format)


@app.command()
def members(
        organization_name: str = organization_name_argument,
        output_format: OutputFormat = output_format_option,
        output: typer.FileTextWrite = output_option,
) -> None:
    started = datetime.now()
    title = f"Members of {organization_name} on github"

    organization = g.get_organization(organization_name)
    member_info = [
        dict(
            login=member.login, name=member.name,
            email=member.email, url=member.html_url,
            membership_state=membership.state, membership_role=membership.role,
        )
        for member, membership in sorted(
            get_members_and_membership(organization),
            key=lambda member_and_membership: member_and_membership[0].login
        )
    ]

    if output_format == OutputFormat.json:
        output_json(member_info, output)

    elif output_format == OutputFormat.html:
        doc, tag, text = Doc().tagtext()
        with tag('html'):
            create_html_head(doc, title)
            with tag("body"):
                with tag("h1"):
                    text(title)
                with tag("table"):
                    with tag("tr"):
                        with tag("th", colspan=4, klass="column-group-header"):
                            text("Member")
                        doc.stag("th")
                        with tag("th", colspan=2, klass="column-group-header"):
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
                    for member in member_info:
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
                                doc.asis(create_html_email_href(email))
                            with tag("td"):
                                doc.asis(create_html_url_href(url))
                            doc.stag("td")
                            with tag("td"):
                                text(member['membership_state'])
                            with tag("td"):
                                text(member['membership_role'])
                create_html_footer(doc, started)
        print(indent(doc.getvalue()), file=output)

    elif output_format == OutputFormat.text:
        table = Table("Member", "Membership state", "Membership role", box=box.SQUARE)
        for member in member_info:
            table.add_row(format_member(member), member['membership_state'], member['membership_role'])
        output_text_elements(title, table, started, output)

    else:
        OutputFormat.unknown(output_format)


if __name__ == "__main__":
    app()
