from __future__ import annotations

from clab.repos.base import RepoAdapter


class XarrayRepoAdapter(RepoAdapter):
    @property
    def runnable(self) -> bool:
        return False
