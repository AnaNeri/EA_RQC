"""Command-line entry for demo package.

Usage:
  python -m demo                # runs circuit_demo by default
  python -m demo.circuit_demo   # run specific demo module
"""
from __future__ import annotations

import sys
import importlib
from typing import List, Tuple


def _discover_demos() -> List[Tuple[str, str]]:
    # Return list of (module_name, short_description)
    modules = [
        "circuit_demo",
        "circuit_demo_entangling",
        "circuit_demo_grover_oracle",
        "circuit_demo_qpc_identities",
        "circuit_demo_qpc_approximation",
        "circuit_demo_qpc_correction_priority",
        "circuit_demo_qpc_protective_scaffold",
        "circuit_demo_qpc_error_correction",
        "circuit_demo_qpc_x_vs_hzh",
        "circuit_demo_quantamorphism_protection",
        "map_demo",
    ]
    demos: List[Tuple[str, str]] = []
    for mod in modules:
        try:
            m = importlib.import_module(f"demo.{mod}")
            doc = (m.__doc__ or "").strip().splitlines()[0] if m.__doc__ else ""
            if hasattr(m, "main"):
                demos.append((mod, doc))
        except Exception:
            # ignore modules that fail to import in this environment
            continue
    return demos


def _print_menu(demos: List[Tuple[str, str]]) -> None:
    print("Available demos:")
    for idx, (mod, doc) in enumerate(demos, start=1):
        label = doc or mod
        print(f"  {idx:02d}. {mod} — {label}")
    print("")


def _run_demo_by_module(name: str) -> None:
    m = importlib.import_module(f"demo.{name}")
    if not hasattr(m, "main"):
        raise RuntimeError(f"Demo module {name} has no main()")
    m.main()


def main(argv: List[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    demos = _discover_demos()
    if not demos:
        print("No demos available (failed to import demo modules).")
        return

    # --list prints menu and exits
    if argv and argv[0] in ("--list", "-l"):
        _print_menu(demos)
        return

    # If user provided module name or index, dispatch non-interactively
    if argv:
        key = argv[0]
        # numeric index (1-based)
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(demos):
                _run_demo_by_module(demos[idx][0])
                return
            else:
                print(f"Invalid demo index: {key}")
                return
        # name or module key
        names = [m for m, _ in demos]
        if key in names:
            _run_demo_by_module(key)
            return
        print(f"Unknown demo '{key}'. Use --list to see available demos.")
        return

    # Interactive menu
    if not sys.stdin or not sys.stdin.isatty():
        # Non-interactive: run default
        _run_demo_by_module(demos[0][0])
        return

    _print_menu(demos)
    try:
        sel = input("Select demo number or name (q to quit): ").strip()
    except EOFError:
        return
    if not sel or sel.lower() in ("q", "quit", "exit"):
        return
    if sel.isdigit():
        idx = int(sel) - 1
        if 0 <= idx < len(demos):
            _run_demo_by_module(demos[idx][0])
            return
        print("Invalid selection")
        return

    # treat as name
    names = [m for m, _ in demos]
    if sel in names:
        _run_demo_by_module(sel)
        return

    print("Unknown selection")


if __name__ == "__main__":
    main()
