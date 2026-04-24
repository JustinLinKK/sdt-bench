from sdt_bench.projects.base import ProjectAdapter
from sdt_bench.schemas.project import ProjectSpec


def get_project_adapter(project_spec: ProjectSpec) -> ProjectAdapter:
    return ProjectAdapter(project_spec=project_spec)


__all__ = ["ProjectAdapter", "get_project_adapter"]
