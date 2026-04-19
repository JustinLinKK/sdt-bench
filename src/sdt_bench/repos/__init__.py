from sdt_bench.repos.base import RepoAdapter
from sdt_bench.repos.pytest_repo import PytestRepoAdapter
from sdt_bench.repos.requests_repo import RequestsRepoAdapter
from sdt_bench.repos.sphinx_repo import SphinxRepoAdapter
from sdt_bench.repos.sqlfluff_repo import SqlfluffRepoAdapter
from sdt_bench.repos.xarray_repo import XarrayRepoAdapter
from sdt_bench.schemas.repo import RepoSpec


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
