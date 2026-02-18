# github-audit
Audit GitHub repositories and members of an organization

## Installation

Clone the repository:

```console
$ git clone https://github.com/ICTU/github-audit.git
```

Create a GitHub token.

Create a `.audit.cfg` file with contents:

```ini
[github.com]
token = <your token>
```
optionally you may specify the organization you work for:
```ini
[github.com]
token = <your token>
organization = <your organization>
```

## Usage

Run the audit script with `--help` for instructions:

```console
$ uv run --script audit.py --help
```

