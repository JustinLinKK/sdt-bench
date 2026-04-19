from clab.repos.base import RepoAdapter
from clab.repos.pytest_repo import PytestRepoAdapter
from clab.repos.requests_repo import RequestsRepoAdapter
from clab.repos.sphinx_repo import SphinxRepoAdapter
from clab.repos.sqlfluff_repo import SqlfluffRepoAdapter
from clab.repos.xarray_repo import XarrayRepoAdapter
from clab.schemas.repo import RepoSpec


def get_repo_adapter(repo_spec: RepoSpec) -> RepoAdapter:
    mapping = {
        "requests": RequestsRepoAdapter,
        "pytest": PytestRepoAdapter,
        "sphinx": SphinxRepoAdapter,
        "sqlfluff": SqlfluffRepoAdapter,
        "xarray": XarrayRepoAdapter,
    }
    adapter_cls = mapping.get(repo_spec.name, RepoAdapter)
    return adapter_cls(repo_spec=repo_spec)


__all__ = ["RepoAdapter", "get_repo_adapter"]
