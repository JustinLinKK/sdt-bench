from __future__ import annotations

from clab.repos.base import RepoAdapter


class PytestRepoAdapter(RepoAdapter):
    @property
    def runnable(self) -> bool:
        return False
