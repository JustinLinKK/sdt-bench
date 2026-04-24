from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sdt_bench.schemas.project import ProjectSpec


@dataclass(slots=True)
class ProjectAdapter:
    project_spec: ProjectSpec

    @property
    def runnable(self) -> bool:
        return True

    def assert_supported(self) -> None:
        if not self.runnable:
            raise NotImplementedError(f"{self.project_spec.project_id} is not runnable in v0")

    def prepare_workspace(self, workspace: Path) -> None:
        del workspace
