from __future__ import annotations

from clab.repos.base import RepoAdapter


class SqlfluffRepoAdapter(RepoAdapter):
    @property
    def runnable(self) -> bool:
        return False
