"""Run Ansatz O from the uploaded homework on a local Qiskit simulator.

Hamiltonian, in the homework's qubit order:

    H = 2 * (Z0 Z1 I2) + (Z0 I1 Z2) - (I0 X1 X2)

The script first finds good theta/phi parameters with exact statevector
expectations, then estimates the energy from finite-shot simulator counts.
"""

from __future__ import annotations

import argparse
import math
import statistics
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


TWO_PI = 2.0 * math.pi


@dataclass(frozen=True)
class Expectations:
    zzi: float
    ziz: float
    ixx: float

    @property
    def energy(self) -> float:
        return 2.0 * self.zzi + self.ziz - self.ixx


def build_ansatz_o(theta: float, phi: float) -> QuantumCircuit:
    """Create Ansatz O: Ry(theta,0), CX(0,1), Ry(phi,2), CX(2,1), CX(2,0)."""
    circuit = QuantumCircuit(3)
    circuit.ry(theta, 0)
    circuit.cx(0, 1)
    circuit.ry(phi, 2)
    circuit.cx(2, 1)
    circuit.cx(2, 0)
    return circuit


def exact_expectations(theta: float, phi: float) -> Expectations:
    """Compute exact expectations in the homework's q2 q1 q0 Pauli order."""
    state = Statevector.from_instruction(build_ansatz_o(theta, phi))

    # Qiskit Pauli labels are written q2 q1 q0, matching the homework notation.
    zzi = state.expectation_value(SparsePauliOp.from_list([("ZZI", 1.0)])).real
    ziz = state.expectation_value(SparsePauliOp.from_list([("ZIZ", 1.0)])).real
    ixx = state.expectation_value(SparsePauliOp.from_list([("IXX", 1.0)])).real
    return Expectations(float(zzi), float(ziz), float(ixx))


def exact_energy(params: np.ndarray) -> float:
    theta, phi = (float(value % TWO_PI) for value in params)
    return exact_expectations(theta, phi).energy


def optimize_parameters(grid_points: int) -> tuple[float, float, float]:
    """Use a coarse grid seed followed by Nelder-Mead minimization."""
    values = np.linspace(0.0, TWO_PI, grid_points, endpoint=False)
    best_params = np.array([0.0, 0.0])
    best_energy = float("inf")

    for theta in values:
        for phi in values:
            energy = exact_energy(np.array([theta, phi]))
            if energy < best_energy:
                best_energy = energy
                best_params = np.array([theta, phi])

    result = minimize(
        exact_energy,
        best_params,
        method="Nelder-Mead",
        options={"maxiter": 1000, "xatol": 1e-12, "fatol": 1e-12},
    )
    theta, phi = (float(value % TWO_PI) for value in result.x)
    return theta, phi, exact_expectations(theta, phi).energy


def expectation_from_counts(counts: dict[str, int], qubits: tuple[int, ...]) -> float:
    """Convert bitstring counts into an expectation for a product observable."""
    total = sum(counts.values())
    weighted_sum = 0

    for bitstring, count in counts.items():
        bits_by_qubit = bitstring.replace(" ", "")[::-1]
        eigenvalue = 1
        for qubit in qubits:
            eigenvalue *= 1 if bits_by_qubit[qubit] == "0" else -1
        weighted_sum += eigenvalue * count

    return weighted_sum / total


def run_counts(circuit: QuantumCircuit, shots: int, seed: int) -> dict[str, int]:
    measured = circuit.copy()
    measured.measure_all()
    simulator = AerSimulator()
    compiled = transpile(measured, simulator)
    result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()
    return result.get_counts(compiled)


def sampled_expectations(theta: float, phi: float, shots: int, seed: int) -> Expectations:
    z_circuit = build_ansatz_o(theta, phi)
    z_counts = run_counts(z_circuit, shots=shots, seed=seed)

    x_circuit = build_ansatz_o(theta, phi)
    x_circuit.h(0)
    x_circuit.h(1)
    x_counts = run_counts(x_circuit, shots=shots, seed=seed + 1)

    return Expectations(
        zzi=expectation_from_counts(z_counts, (2, 1)),
        ziz=expectation_from_counts(z_counts, (2, 0)),
        ixx=expectation_from_counts(x_counts, (1, 0)),
    )


def format_cell(value: float) -> str:
    return f"({value:.6f})"


def print_outcome_table(samples: list[Expectations]) -> None:
    """Print the five-run report table with operators as rows."""
    energies = [sample.energy for sample in samples]
    mean_energy = statistics.mean(energies)
    variance_energy = statistics.variance(energies) if len(energies) > 1 else 0.0
    std_energy = statistics.stdev(energies) if len(energies) > 1 else 0.0
    rows = [
        ["Exp", "{IXX}", *[format_cell(sample.ixx) for sample in samples]],
        ["", "{ZZI}", *[format_cell(sample.zzi) for sample in samples]],
        ["", "{ZIZ}", *[format_cell(sample.ziz) for sample in samples]],
        ["E", "E=2{ZZI}+{ZIZ}-{IXX}", *[format_cell(energy) for energy in energies]],
        [
            "Stats",
            f"<E>={mean_energy:.6f}  Var={variance_energy:.6f}  sigma={std_energy:.6f}",
            *["" for _ in samples],
        ],
    ]
    headers = ["", "Operator", *[f"Run {index + 1}" for index in range(len(samples))]]
    widths = [
        max(len(str(row[column])) for row in [headers, *rows])
        for column in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"

    print("Simulation Outcomes -- Ansatz O")
    print(render(headers))
    print(separator)
    for row in rows:
        print(render(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run homework Ansatz O with Qiskit Aer.")
    parser.add_argument("--theta", type=float, help="Use this theta instead of optimizing.")
    parser.add_argument("--phi", type=float, help="Use this phi instead of optimizing.")
    parser.add_argument("--shots", type=int, default=1024, help="Simulator shots per run.")
    parser.add_argument("--runs", type=int, default=5, help="Number of sampled runs.")
    parser.add_argument("--grid-points", type=int, default=41, help="Coarse optimizer grid size.")
    parser.add_argument("--seed", type=int, default=20260515, help="Base simulator seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (args.theta is None) != (args.phi is None):
        raise SystemExit("Provide both --theta and --phi, or neither.")

    if args.theta is None:
        theta, phi, exact = optimize_parameters(args.grid_points)
    else:
        theta, phi = args.theta % TWO_PI, args.phi % TWO_PI
        exact = exact_expectations(theta, phi).energy

    exact_values = exact_expectations(theta, phi)
    circuit = build_ansatz_o(theta, phi)

    print("Ansatz O circuit:")
    print(circuit.draw(output="text"))
    print(f"theta = {theta:.12f} rad")
    print(f"phi   = {phi:.12f} rad")
    print("Exact statevector expectations:")
    print(f"  <ZZI> = {exact_values.zzi:.12f}")
    print(f"  <ZIZ> = {exact_values.ziz:.12f}")
    print(f"  <IXX> = {exact_values.ixx:.12f}")
    print(f"  E     = {exact:.12f}")
    print()
    print(f"Sampled simulator runs ({args.shots} shots each):")

    samples: list[Expectations] = []
    for run_index in range(args.runs):
        values = sampled_expectations(
            theta=theta,
            phi=phi,
            shots=args.shots,
            seed=args.seed + 2 * run_index,
        )
        samples.append(values)

    if samples:
        print_outcome_table(samples)


if __name__ == "__main__":
    main()
