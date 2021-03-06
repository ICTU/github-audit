from typing import Any, Dict

import github.GithubObject


class CodeScanAlertInstanceLocation(github.GithubObject.NonCompletableGithubObject):
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    @property
    def path(self) -> str: ...
    @property
    def start_line(self) -> int: ...
    @property
    def start_column(self): -> int: ...
    @property
    def end_line(self) -> int: ...
    @property
    def end_column(self) -> int: ...

    def _initAttributes(self) -> None: ...
    def _useAttributes(self, attributes: Dict[str, Any]) -> None: ...
