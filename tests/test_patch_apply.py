from __future__ import annotations

from pathlib import Path

from sdt_bench.env.patching import apply_patch_text, measure_patch
from sdt_bench.utils.subprocess import run_command


def test_patch_apply_and_measure(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "demo.txt").write_text("before\n", encoding="utf-8")
    run_command(["git", "init"], cwd=repo)
    run_command(["git", "config", "user.email", "sdt-bench@example.com"], cwd=repo)
    run_command(["git", "config", "user.name", "sdt-bench"], cwd=repo)
    run_command(["git", "add", "."], cwd=repo)
    run_command(["git", "commit", "-m", "init"], cwd=repo)

    patch = """diff --git a/demo.txt b/demo.txt
index 4f87e0b..65b2df8 100644
--- a/demo.txt
+++ b/demo.txt
@@ -1 +1 @@
-before
+after
"""
    applied, error = apply_patch_text(repo, patch)
    assert applied is True
    assert error is None
    files_changed, lines_added, lines_removed = measure_patch(repo)
    assert files_changed == 1
    assert lines_added == 1
    assert lines_removed == 1
