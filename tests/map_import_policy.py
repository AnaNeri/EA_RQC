from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_canonical_namespaces_do_not_import_legacy_modules():
    # Keep legacy imports confined to compatibility layers and tests.
    checked_roots = [
        ROOT / "evolution",
        ROOT / "adapters",
        ROOT / "demo",
    ]

    violations: list[str] = []
    for folder in checked_roots:
        for file in folder.rglob("*.py"):
            content = file.read_text(encoding="utf-8")
            if "from map." in content or "import map." in content or "from parser." in content or "import parser." in content:
                violations.append(str(file.relative_to(ROOT)))

    assert not violations, f"Legacy imports found in canonical code: {violations}"
