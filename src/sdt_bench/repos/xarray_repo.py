from __future__ import annotations

from sdt_bench.repos.base import RepoAdapter


class XarrayRepoAdapter(RepoAdapter):
    @property
    def runnable(self) -> bool:
        return False
