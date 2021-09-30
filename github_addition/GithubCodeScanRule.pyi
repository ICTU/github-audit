from typing import Any, Dict

import github.GithubObject


class CodeScanRule(github.GithubObject.NonCompletableGithubObject):
    def __repr__(self) -> str: ...
    @property
    def id(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def severity(self) -> str: ...
    @property
    def security_severity_level(self) -> str: ...
    @property
    def description(self) -> str: ...

    def _initAttributes(self) -> None: ...
    def _useAttributes(self, attributes: Dict[str, Any]) -> None: ...