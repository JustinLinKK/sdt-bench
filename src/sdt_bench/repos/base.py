from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdt_bench.schemas.repo import RepoSpec


@dataclass(slots=True)
class RepoAdapter:
    repo_spec: RepoSpec

    @property
    def runnable(self) -> bool:
        return True

    def assert_supported(self) -> None:
        if not self.runnable:
            raise NotImplementedError(f"{self.repo_spec.name} is not runnable in v0")

    def prepare_workspace(self, workspace: Path) -> None:
        del workspace
