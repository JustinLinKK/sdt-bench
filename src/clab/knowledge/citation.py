from __future__ import annotations


def citation_overlap(citations: list[str], expected_paths: list[str]) -> float:
    if not citations or not expected_paths:
        return 0.0
    expected = set(expected_paths)
    cited = set(citations)
    return len(expected & cited) / len(expected)
