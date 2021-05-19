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

Install the requirements:

```console
$ python3 -m venv venv
$ . venv/bin/activate
$ pip install wheel
$ pip install -r requirements.txt
```

## Usage

Run the audit script with `--help` for instructions:

```console
$ python3 audit.py --help
```

