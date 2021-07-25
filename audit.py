"""Audit GitHub organizations."""

import configparser
import json
from enum import Enum
from typing import Optional, Tuple, Generator
from datetime import datetime

from github import Github
from github.Membership import Membership
from github.NamedUser import NamedUser
from github.Organization import Organization
from github.Repository import Repository

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


class ReportBase:

    def __init__(self, started, output):
        self.started = started
        self.output = output

    def begin_report(self, title: str):
        raise NotImplementedError(f"{self.__class__.__name__}.begin_report()")

    def end_report(self):
        raise NotImplementedError(f"{self.__class__.__name__}.end_report()")

    def empty_line(self):
        raise NotImplementedError(f"{self.__class__.__name__}.empty_line()")

    def begin_table(self):
        raise NotImplementedError(f"{self.__class__.__name__}.begin_table()")

    def table_row(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.table_row()")

    def end_table(self):
        raise NotImplementedError(f"{self.__class__.__name__}.end_table()")


class TextReportBase(ReportBase):

    table = None

    def __init__(self, started, output):
        super().__init__(started, output)
        self.console = Console(file=output)

    def begin_report(self, title: str):
        self.console.print(title)

    def end_report(self):
        self.console.print(format_generated_timestamp(self.started))

    def empty_line(self):
        self.console.print()

    def begin_table(self):
        if self.table is not None:
            raise RuntimeError("already building a table")
        self.table = Table(box=box.SQUARE)
        self._table_header()

    def _table_header(self):
        raise NotImplementedError(f"{self.__class__.__name__}._table_header()")

    def table_row(self, *args, **kwargs):
        self._table_row(*args, **kwargs)

    def _table_row(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.table_row()")

    def end_table(self):
        if self.table is None:
            raise RuntimeError("not building a table")
        self.console.print()
        self.console.print(self.table)
        self.console.print()


class HtmlReportBase(ReportBase):

    def __init__(self, started, output):
        super().__init__(started, output)
        doc, tag, text = Doc().tagtext()
        self.doc = doc
        self.tag = tag
        self.text = text

    def begin_report(self, title: str):
        self.doc.asis("<html>")
        with self.tag("head"):
            self.doc.asis('<meta charset="UTF-8">')
            with self.tag("title"):
                self.text(title)
            with self.tag("style"):
                self.text("\nbody {font-size: 100%; }")
                self.text("\ndiv.footer {font-size: 50%; padding-top: 24px; }")
                self.text("\ntable {border-spacing: 0; border-collapse: collapse; }")
                self.text("\nth {vertical-align: bottom; }")
                self.text("\ntd {vertical-align: top; }")
                self.text("\n.centered {text-align: center; }")
                self.text("\n.column-group-header {border-bottom: 1px solid black; }")
                self.text("\n.first-row {border-top: 1px solid black; }")
                self.text("\n.right {text-align: right; }")
        self.doc.asis("<body>")
        with self.tag("h1"):
            self.text(title)
        self._prologue()
        self._begin_main()

    def end_report(self):
        self._end_main()
        self._epilogue()
        with self.doc.tag("div", klass="footer"):
            self.doc.text(format_generated_timestamp(self.started))
        self.doc.asis("</body>")
        self.doc.asis("</html>")
        print(indent(self.doc.getvalue()), file=self.output)

    def _prologue(self):
        pass

    def _epilogue(self):
        pass

    def _command_options(self, options):
        with self.tag("h2"):
            self.text("Options")
        with self.tag("table"):
            for name, value in options.items():
                with self.tag("tr"):
                    with self.tag("td"):
                        self.text(name)
                    with self.tag("td"):
                        self.text(value)

    # main part of the report

    def _begin_main(self):
        with self.tag("h2"):
            self.text("Report")

    def _end_main(self):
        pass

    # report content

    def empty_line(self):
        self.doc.stag("br")

    # table creation

    def begin_table(self):
        self.doc.asis("<table>")
        self._table_header()

    def _table_header(self):
        raise NotImplementedError(f"{self.__class__.__name__}._table_header()")

    def table_row(self, *args, **kwargs):
        with self.tag("tr"):
            self._table_row(*args, **kwargs)

    def _table_row(self, *args, **kwargs):
        raise NotImplementedError(f"{self.__class__.__name__}.__table_row()")

    def end_table(self):
        self.doc.asis("</table>")


# Github API wrappers

def get_repos(
        organization: Organization,
        include_forked_repositories: bool,
        include_archived_repositories: bool
) -> Generator[Repository, None, None]:
    """Return the repositories of the organization, displaying a scrollbar."""
    with typer.progressbar(organization.get_repos(), length=organization.public_repos, label="Collecting") as repos:
        for repo in repos:
            if repo.archived and not include_archived_repositories:
                continue
            if repo.fork and not include_forked_repositories:
                continue
            yield repo


def get_contributors(repo: Repository) -> Generator[NamedUser, None, None]:
    """Return the non-bot contributors to a repository"""
    for contributor in repo.get_contributors():
        if contributor.type.lower() != "bot":
            yield contributor


def get_members_and_membership(organization) -> Generator[Tuple[NamedUser, Membership], None, None]:
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
    dt_as_text = dt.astimezone().strftime('%c %Z')
    return f"generated on {dt_as_text}"


def format_member(member: dict) -> str:
    """Return the member name and/or login."""
    name = member.get("name") or ""
    login = member.get("login") or ""
    return f"{name} ({login})" if name and login else name or login


# HTML utilities

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


# output creation

def output_json(json_data: list, output: Optional[typer.FileTextWrite]) -> None:
    """
    Output the json data
    :param json_data: data to convert to JSON
    :param output: output file to write to (default: stdout)
    """
    typer.echo(json.dumps(json_data, indent="  "), file=output)


###
# APPLICATION COMMANDS

g = Github(config["github.com"]["token"])
app = typer.Typer()


class RepoTextReport(TextReportBase):

    def __init__(self, started, output, include_archived_repositories, include_forked_repositories):
        super().__init__(started, output)
        self.include_archived_repositories = include_archived_repositories
        self.include_forked_repositories = include_forked_repositories

    def _table_header(self):
        if self.table is None:
            raise RuntimeError("not building a table")
        self.table.add_column("Name")
        if self.include_archived_repositories:
            self.table.add_column("Archived", justify="center")
        if self.include_forked_repositories:
            self.table.add_column("Fork", justify="center")
        self.table.add_column("Pushed at")
        self.table.add_column("Pull request title")
        self.table.add_column("Created at")

    def _table_row(
            self,
            repo_name=None, archived=None, fork=None, pushed_at=None, title=None, created_at=None,
            rowspan=0, first=False
    ):
        if self.table is None:
            raise RuntimeError("not building a table")
        row = []
        row.append(repo_name)
        if self.include_archived_repositories:
            row.append(format_bool(archived) if archived is not None else None)
        if self.include_forked_repositories:
            row.append(format_bool(fork) if fork is not None else None)
        row.append(f"{format_friendly_timestamp(pushed_at, self.started)}" if pushed_at is not None else None)
        if title and created_at:
            row.extend([title, f"{format_friendly_timestamp(created_at, self.started)}"])
        self.table.add_row(*row)


class RepoHtmlReport(HtmlReportBase):

    def __init__(self, started, output, include_archived_repositories, include_forked_repositories):
        super().__init__(started, output)
        self.include_archived_repositories = include_archived_repositories
        self.include_forked_repositories = include_forked_repositories

    def _epilogue(self):
        options = {
            "Archived repositories": "included" if self.include_archived_repositories else "not included",
            "Forked repositories": "included" if self.include_forked_repositories else "not included",
        }
        self._command_options(options)

    def _table_header(self):
        with self.tag("tr"):
            with self.tag("th", rowspan=2):
                self.text("Name")
            if self.include_archived_repositories:
                with self.tag("th", rowspan=2):
                    self.text("Archived")
            if self.include_forked_repositories:
                with self.tag("th", rowspan=2):
                    self.text("Fork")
            with self.tag("th", rowspan=2):
                self.text("Pushed at")
            with self.tag("th", colspan=2, klass="column-group-header"):
                self.text("Pull request")
        with self.tag("tr"):
            with self.tag("th"):
                self.text("Title")
            with self.tag("th"):
                self.text("Created at")

    def _table_row(
            self,
            repo_name=None, archived=None, fork=None, pushed_at=None, title=None, created_at=None,
            rowspan=0, first=False
    ):
        klass_first_row = {"klass": "first-row"} if first else {}
        klass_first_row_centered = {"klass": "first-row centered"} if first else {}
        if repo_name is not None:
            with self.tag("td", **klass_first_row, rowspan=rowspan):
                self.text(repo_name)
        if archived is not None and self.include_archived_repositories:
            with self.tag("td", **klass_first_row_centered, rowspan=rowspan):
                self.doc.asis(format_bool(archived))
        if fork is not None and self.include_forked_repositories:
            with self.tag("td", **klass_first_row_centered, rowspan=rowspan):
                self.doc.asis(format_bool(fork))
        if pushed_at is not None:
            with self.tag("td", **klass_first_row, rowspan=rowspan):
                self.text(format_friendly_timestamp(pushed_at, self.started))
        if title and created_at:
            with self.tag("td", **klass_first_row):
                self.text(title)
            with self.tag("td", **klass_first_row):
                self.text(format_friendly_timestamp(created_at, self.started))
        else:
            self.doc.stag("td", **klass_first_row, colspan=2)


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
        return

    report_class = {
        OutputFormat.html: RepoHtmlReport,
        OutputFormat.text: RepoTextReport,
    }.get(output_format)
    if report_class is None:
        OutputFormat.unknown(output_format)
        return

    report = report_class(started, output, include_archived_repositories, include_forked_repositories)
    report.begin_report(title)
    report.begin_table()
    for repo in sorted(repositories, key=lambda x: x['name'].lower()):
        open_pull_requests = repo["open_pull_requests"]
        columns = [
            repo["name"],
            repo["archived"],
            repo["fork"],
            repo["pushed_at"],
            open_pull_requests[0]["title"] if open_pull_requests else "",
            open_pull_requests[0]["created_at"] if open_pull_requests else "",
        ]
        report.table_row(*columns, rowspan=len(open_pull_requests), first=True)
        for pr in open_pull_requests[1:]:
            report.table_row(title=pr["title"], created_at=pr["created_at"])
    report.end_table()
    report.end_report()


class RepoContributionsTextReport(TextReportBase):

    def __init__(self, started, output, include_archived_repositories, include_forked_repositories):
        super().__init__(started, output)
        self.include_archived_repositories = include_archived_repositories
        self.include_forked_repositories = include_forked_repositories

    def _table_header(self):
        if self.table is None:
            raise RuntimeError("not building a table")

        self.table.add_column("Name")
        self.table.add_column("Contributor")
        self.table.add_column("Nr. of contributions", justify="right")

    def _table_row(self, repo_name=None, contributor=None, rowspan=0, first=False):
        if self.table is None:
            raise RuntimeError("not building a table")
        row = []
        row.append(repo_name)
        if contributor is not None:
            row.append(format_member(contributor))
            row.append(format_int(contributor.get("contributions")))
        else:
            row.extend((None, None))
        self.table.add_row(*row)


class RepoContributionsHtmlReport(HtmlReportBase):

    def __init__(self, started, output, include_archived_repositories, include_forked_repositories):
        super().__init__(started, output)
        self.include_archived_repositories = include_archived_repositories
        self.include_forked_repositories = include_forked_repositories

    def _epilogue(self):
        options = {
            "Archived repositories": "included" if self.include_archived_repositories else "not included",
            "Forked repositories": "included" if self.include_forked_repositories else "not included",
        }
        self._command_options(options)

    def _table_header(self):
        with self.tag("tr"):
            with self.tag("th", rowspan=2):
                self.text("Name")
            with self.tag("th", colspan=5, klass="column-group-header"):
                self.text("Contributor")
        with self.tag("tr"):
            with self.tag("th"):
                self.text("Name")
            with self.tag("th"):
                self.text("Login")
            with self.tag("th"):
                self.text("Email")
            with self.tag("th"):
                self.text("Profile")
            with self.tag("th"):
                self.text("#contributions")

    def _table_row(self, repo_name=None, contributor=None, rowspan=0, first=False):
        klass_first_row = {"klass": "first-row"} if first else {}
        klass_first_row_right = {"klass": "first-row right"} if first else {"klass": "right"}
        if repo_name is not None:
            with self.tag("td", **klass_first_row, rowspan=rowspan):
                self.text(repo_name)
        if contributor is None:
            contributor = {}
        with self.tag("td", **klass_first_row):
            self.text(contributor.get("name") or "")
        with self.tag("td", **klass_first_row):
            self.text(contributor.get("login") or "")
        with self.tag("td", **klass_first_row):
            self.doc.asis(create_html_email_href(contributor.get("email")) or "")
        with self.tag("td", **klass_first_row):
            self.doc.asis(create_html_url_href(contributor.get("url")) or "")
        with self.tag("td", **klass_first_row_right):
            self.text(format_int(contributor.get("contributions")))


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
        return

    report_class = {
        OutputFormat.html: RepoContributionsHtmlReport,
        OutputFormat.text: RepoContributionsTextReport,
    }.get(output_format)
    if report_class is None:
        OutputFormat.unknown(output_format)
        return

    report = report_class(started, output, include_archived_repositories, include_forked_repositories)
    report.begin_report(title)
    report.begin_table()
    for repo in sorted(repositories, key=lambda x: x['name'].lower()):
        contributors = list(sorted(
            repo['contributors'],
            key=lambda x: (1_000_000_000 - x['contributions'], x['login'].lower())
        ))
        if len(contributors) == 0:
            contributors = [{}]
        report.table_row(repo_name=repo["name"], contributor=contributors[0], first=True, rowspan=len(contributors))
        for contributor in contributors[1:]:
            report.table_row(contributor=contributor)
    report.end_table()
    report.end_report()


class MembersTextReport(TextReportBase):

    def _table_header(self):
        if self.table is None:
            raise RuntimeError("not building a table")
        self.table.add_column("Member")
        self.table.add_column("Membership state")
        self.table.add_column("Membership role")

    def _table_row(self, member):
        if self.table is None:
            raise RuntimeError("not building a table")
        self.table.add_row(format_member(member), member['membership_state'], member['membership_role'])


class MembersHtmlReport(HtmlReportBase):

    def _table_header(self):

        with self.tag("tr"):
            with self.tag("th", colspan=4, klass="column-group-header"):
                self.text("Member")
            self.doc.stag("th")
            with self.tag("th", colspan=2, klass="column-group-header"):
                self.text("Membership")
        with self.tag("tr"):
            with self.tag("th"):
                self.text("Name")
            with self.tag("th"):
                self.text("Login")
            with self.tag("th"):
                self.text("Email")
            with self.tag("th"):
                self.text("Profile")
            self.doc.stag("th")
            with self.tag("th"):
                self.text("state")
            with self.tag("th"):
                self.text("role")

    def _table_row(self, member):
        email = member.get("email")
        with self.tag("td"):
            self.text(member.get("name") or "")
        with self.tag("td"):
            self.text(member.get("login") or "")
        with self.tag("td"):
            self.doc.asis(create_html_email_href(email) if email else "")
        with self.tag("td"):
            self.doc.asis(create_html_url_href(member.get("url") or ""))
        self.doc.stag("td")
        with self.tag("td"):
            self.text(member['membership_state'])
        with self.tag("td"):
            self.text(member['membership_role'])


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
        return

    report_class = {
        OutputFormat.html: MembersHtmlReport,
        OutputFormat.text: MembersTextReport,
    }.get(output_format)
    if report_class is None:
        OutputFormat.unknown(output_format)
        return

    report = report_class(started, output)
    report.begin_report(title)
    report.begin_table()
    for member in member_info:
        report.table_row(member)
    report.end_table()
    report.end_report()


if __name__ == "__main__":
    app()
