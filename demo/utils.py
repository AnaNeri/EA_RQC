from __future__ import annotations

import importlib
from dataclasses import dataclass

from adapters.qiskit import parse_qiskit_backend_gate_catalog, parse_qiskit_backend_noise_model


@dataclass(frozen=True)
class DemoNoiseModel:
    qubit_error_rates: dict[int, float]


def _instantiate_fake_backend(backend_name: str):
    try:
        module = importlib.import_module("qiskit_ibm_runtime.fake_provider")
    except ModuleNotFoundError:
        module = importlib.import_module("qiskit.providers.fake_provider")

    backend_class = getattr(module, backend_name, None)
    if backend_class is None:
        raise ValueError(f"Unknown fake backend class: {backend_name}")
    return backend_class()


def _normalize_gate_name(name: str) -> str | None:
    raw = str(name).lower().strip()
    if not raw:
        return None

    supported = {
        "id", "x", "y", "z", "h", "sx", "rx", "ry", "rz", "s", "sdg", "t", "tdg", "p", "phase",
        "u", "cx", "cz", "ecr", "swap", "cy", "ch",
    }
    if raw in supported:
        return raw

    # Backend targets can expose qubit-scoped aliases like id126, sx87, rz12.
    for prefix in ("id", "sx", "rz", "x"):
        suffix = raw[len(prefix) :]
        if raw.startswith(prefix) and suffix.isdigit():
            return prefix

    return None


def _sanitize_gate_catalog(gate_catalog: list[str]) -> list[str]:
    sanitized: list[str] = []
    for name in gate_catalog:
        normalized = _normalize_gate_name(name)
        if normalized is None:
            continue
        if normalized not in sanitized:
            sanitized.append(normalized)
    return sanitized


def load_fake_kyiv_device_data(device_qubits: int) -> tuple[DemoNoiseModel, list[str], str]:
    candidates = ("FakeKyiv",)
    last_error: Exception | None = None

    for backend_name in candidates:
        try:
            backend = _instantiate_fake_backend(backend_name)
            parsed_noise = parse_qiskit_backend_noise_model(backend)
            gate_catalog = _sanitize_gate_catalog(parse_qiskit_backend_gate_catalog(backend))
            rates = parsed_noise.qubit_error_rates

            # determine actual device size, supporting multiple qiskit provider versions
            total_qubits = getattr(backend, "num_qubits", None)
            if total_qubits is None:
                try:
                    cfg = backend.configuration()
                    total_qubits = getattr(cfg, "n_qubits", getattr(cfg, "num_qubits", None))
                except Exception:
                    total_qubits = device_qubits

            try:
                total_int = int(total_qubits)
            except Exception:
                total_int = device_qubits

            # pick the first N physical qubits (0..N-1), capped by the actual device size
            selected_n = min(device_qubits, total_int)
            selected_rates = {
                qubit: float(rates.get(qubit, rates.get(str(qubit), 0.0)))
                for qubit in range(selected_n)
            }
            return DemoNoiseModel(qubit_error_rates=selected_rates), gate_catalog, backend_name
        except Exception as exc:
            last_error = exc

    raise RuntimeError(
        "Could not load FakeKyiv backend from qiskit fake providers. "
        "Ensure qiskit-ibm-runtime with FakeKyiv is installed."
    ) from last_error


def localize_circuit(circuit):
    """Return a localized copy of `circuit` where logical qubits are re-indexed
    starting at 0. Useful before rendering matrices or images.
    """
    # Import lazily to avoid circular imports
    from circuit.entities.circuit_list import CircuitGate, CircuitList

    if circuit.cluster:
        logical_qubits = list(circuit.cluster)
    else:
        logical_qubits = sorted({q for gate in circuit.gates for q in gate.qubits})

    if not logical_qubits:
        return circuit.clone()

    qubit_map = {qubit: idx for idx, qubit in enumerate(logical_qubits)}
    localized_gates = []
    for gate in circuit.gates:
        localized_gates.append(
            CircuitGate.from_values(
                name=gate.name,
                qubits=tuple(qubit_map.get(q, q) for q in gate.qubits),
                parameters=gate.parameters,
                depth=gate.depth,
                immutable=gate.immutable,
            )
        )

    return CircuitList(
        gates=localized_gates,
        cluster=tuple(range(len(logical_qubits))),
        max_depth=circuit.max_depth,
        max_gates=circuit.max_gates,
    )


def circuit_to_png_bytes(circuit) -> bytes:
    """Render a simple visualization of a `CircuitList` to PNG bytes using
    matplotlib. This draws horizontal qubit lines and gate boxes positioned
    by gate depth.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib is required for circuit image rendering")

    from io import BytesIO

    localized = localize_circuit(circuit)
    # number of logical qubits
    nq = 0
    for g in localized.gates:
        nq = max(nq, max(g.qubits) if g.qubits else 0)
    # safer: compute max qubit index
    max_q = 0
    for g in localized.gates:
        if g.qubits:
            max_q = max(max_q, max(g.qubits))
    nq = max_q + 1 if localized.gates else 1

    # depth positions
    positions = [g.depth for g in localized.gates]
    if positions:
        max_depth = max(positions) + 1
    else:
        max_depth = 1

    fig_height = max(2.0, 0.6 * nq)
    fig_width = max(4.0, 0.6 * max_depth)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # draw qubit lines
    y_coords = list(range(nq))[::-1]
    for i, y in enumerate(y_coords):
        ax.hlines(y, 0, max_depth + 1, color="black", linewidth=1)
        ax.text(-0.3, y, f"q{i}", va="center", ha="right", fontsize=10)

    # draw gates
    for g in localized.gates:
        depth = g.depth
        if not g.qubits:
            continue
        if len(g.qubits) == 1:
            q = g.qubits[0]
            y = y_coords[q]
            rect = plt.Rectangle((depth - 0.4, y - 0.2), 0.8, 0.4, facecolor="#66c2a5", edgecolor="black")
            ax.add_patch(rect)
            ax.text(depth, y, g.name, ha="center", va="center", fontsize=8)
        else:
            # multi-qubit gate: draw box spanning qubits and connecting lines
            qubits = sorted(g.qubits)
            ys = [y_coords[q] for q in qubits]
            ymin, ymax = min(ys), max(ys)
            rect = plt.Rectangle((depth - 0.4, ymin - 0.2), 0.8, (ymax - ymin) + 0.4, facecolor="#8da0cb", edgecolor="black")
            ax.add_patch(rect)
            ax.text(depth, (ymin + ymax) / 2, g.name, ha="center", va="center", fontsize=8)
            # vertical connector
            ax.vlines(depth, ymin, ymax, color="black", linewidth=1)

    ax.set_xlim(-1, max_depth + 1)
    ax.set_ylim(-1, nq)
    ax.axis("off")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def coupling_graph_to_png_bytes(edges: dict[tuple[int, int], float], top_clusters: list[list[int]] | None = None) -> bytes:
    """Render a coupling graph (edges dict) to PNG bytes. Optionally highlight top_clusters.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        raise RuntimeError("matplotlib is required for graph image rendering")

    from io import BytesIO
    # Nodes set
    nodes = set()
    for (u, v) in edges:
        nodes.add(u)
        nodes.add(v)
    if not nodes:
        nodes = {0}

    nodes = sorted(nodes)
    n = len(nodes)

    # positions on circle
    import math

    pos = {}
    for idx, node in enumerate(nodes):
        angle = 2 * math.pi * idx / n
        pos[node] = (math.cos(angle), math.sin(angle))

    fig, ax = plt.subplots(figsize=(6, 6))
    # draw edges
    for (u, v), w in edges.items():
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        ax.plot([x1, x2], [y1, y2], color="#999999", linewidth=max(0.8, 3.0 * (1.0 - min(w, 0.1))))

    # highlight clusters using blue shades and build legend
    highlight: dict[int, str] = {}
    legend_patches = []
    if top_clusters:
        # choose a Blues colormap and sample descending shades
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors
        from matplotlib.patches import Patch

        cmap = cm.get_cmap("Blues")
        max_show = min(len(top_clusters), 6)
        for rank, cluster in enumerate(top_clusters[:max_show]):
            # sample from the colormap: darker for top rank
            t = 0.9 - (rank * 0.12)
            rgba = cmap(max(0.0, t))
            color = mcolors.to_hex(rgba)
            for q in cluster:
                highlight[q] = color
            legend_patches.append(Patch(facecolor=color, edgecolor="black", label=f"Top {rank+1} cluster"))
        # legend entry for other / unassigned nodes
        legend_patches.append(Patch(facecolor="#cccccc", edgecolor="black", label="Other qubits"))

    # draw nodes
    for node in nodes:
        x, y = pos[node]
        col = highlight.get(node, "#7fb3ff")
        circle = plt.Circle((x, y), 0.08, color=col, ec="black")
        ax.add_patch(circle)
        ax.text(x, y - 0.14, f"q{node}", ha="center", fontsize=8)

    # draw legend if we prepared patches
    if legend_patches:
        ax.legend(handles=legend_patches, loc="lower left", bbox_to_anchor=(0.01, -0.05), frameon=False)

    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    buf = BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()
