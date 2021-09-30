from typing import Any, Dict

import github.GithubObject


class CodeScanTool(github.GithubObject.NonCompletableGithubObject):
    def __repr__(self) -> str: ...
    @property
    def name(self): -> str: ...
    @property
    def version(self): -> str: ...
    @property
    def guid(self): -> str: ...

    def _initAttributes(self) -> None: ...
    def _useAttributes(self, attributes: Dict[str, Any]) -> None: ...