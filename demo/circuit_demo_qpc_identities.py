"""Demo: Numerical verification of QPC algebraic identities.

Shows that the probabilistic combinator identities hold numerically
for channel superoperators built from single-qubit unitaries.
"""

from __future__ import annotations

import numpy as np

from evolution.circuit.matrix_eval import compose_superoperators, mix_channels, unitary_to_superoperator


TOL = 1e-12


def _channel_from_unitary(unitary: np.ndarray) -> np.ndarray:
    return unitary_to_superoperator(np.asarray(unitary, dtype=np.complex128))


def _fro_norm(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, "fro"))


def _print_identity(name: str, lhs: np.ndarray, rhs: np.ndarray) -> bool:
    residual = _fro_norm(lhs, rhs)
    ok = residual <= TOL
    status = "PASS" if ok else "FAIL"
    print(f"{name:<34} | residual={residual:.3e} | {status}")
    return ok


def main() -> None:
    print("=" * 98)
    print("QPC Demo: Probabilistic-Combinator Identities")
    print("=" * 98)

    x = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    h = (1.0 / np.sqrt(2.0)) * np.array([[1, 1], [1, -1]], dtype=np.complex128)
    z = np.array([[1, 0], [0, -1]], dtype=np.complex128)

    u = _channel_from_unitary(x)
    v = _channel_from_unitary(h)
    w = _channel_from_unitary(z)

    p = 0.37
    q = 0.61

    print(f"Using p={p:.2f}, q={q:.2f}, tolerance={TOL:.1e}")
    print("-" * 98)

    results = []
    results.append(
        _print_identity(
            "Unique behavior",
            mix_channels(u, u, p),
            u,
        )
    )
    results.append(
        _print_identity(
            "Absence of noise",
            mix_channels(u, v, 0.0),
            v,
        )
    )
    results.append(
        _print_identity(
            "Dominance of noise",
            mix_channels(u, v, 1.0),
            u,
        )
    )
    results.append(
        _print_identity(
            "Rearrange combinator",
            mix_channels(u, v, p),
            mix_channels(v, u, 1.0 - p),
        )
    )
    results.append(
        _print_identity(
            "Distributive law",
            mix_channels(u, mix_channels(v, w, q), p),
            mix_channels(mix_channels(u, v, p), mix_channels(u, w, p), q),
        )
    )
    results.append(
        _print_identity(
            "Choice-fusion right",
            compose_superoperators(mix_channels(u, v, p), w),
            mix_channels(compose_superoperators(u, w), compose_superoperators(v, w), p),
        )
    )
    results.append(
        _print_identity(
            "Choice-fusion left",
            compose_superoperators(w, mix_channels(u, v, p)),
            mix_channels(compose_superoperators(w, u), compose_superoperators(w, v), p),
        )
    )

    print("-" * 98)
    if all(results):
        print("All QPC identity checks passed.")
    else:
        print("One or more QPC identity checks failed.")


if __name__ == "__main__":
    main()