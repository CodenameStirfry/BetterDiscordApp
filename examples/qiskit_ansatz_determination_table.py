"""Rank homework ansaetze A-O by optimized Hamiltonian energy.

Hamiltonian, in the homework's q2 q1 q0 order:

    H = 2 * ZZI + ZIZ - IXX

For each ansatz this script:
1. optimizes theta/phi with exact statevector expectations;
2. independently samples ZZI, ZIZ, and IXX on the Aer simulator;
3. prints a determination table ranked by the exact optimized energy.
"""

from __future__ import annotations

import argparse
import math
import statistics
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import SparsePauliOp, Statevector
from qiskit_aer import AerSimulator
from scipy.optimize import minimize


TWO_PI = 2.0 * math.pi
BITSTRINGS = tuple(format(index, "03b") for index in range(8))


@dataclass(frozen=True)
class Expectations:
    zzi: float
    ziz: float
    ixx: float

    @property
    def energy(self) -> float:
        return 2.0 * self.zzi + self.ziz - self.ixx


@dataclass(frozen=True)
class Ansatz:
    label: str
    description: str
    parameter_names: tuple[str, ...]
    builder: Callable[[tuple[float, ...]], QuantumCircuit]


@dataclass(frozen=True)
class RankingResult:
    ansatz: Ansatz
    parameters: tuple[float, ...]
    exact_expectations: Expectations
    sampled_energies: tuple[float, ...]

    @property
    def exact_energy(self) -> float:
        return self.exact_expectations.energy

    @property
    def sampled_mean(self) -> float:
        return statistics.mean(self.sampled_energies)

    @property
    def sampled_variance(self) -> float:
        if len(self.sampled_energies) < 2:
            return 0.0
        return statistics.variance(self.sampled_energies)

    @property
    def sampled_sigma(self) -> float:
        if len(self.sampled_energies) < 2:
            return 0.0
        return statistics.stdev(self.sampled_energies)


def circuit() -> QuantumCircuit:
    return QuantumCircuit(3)


def ansatz_a(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.ry(theta, 0)
    qc.cx(0, 1)
    return qc


def ansatz_b(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.h(0)
    qc.rz(theta, 0)
    qc.cx(0, 1)
    return qc


def ansatz_c(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.x(1)
    qc.rx(theta, 0)
    qc.cx(0, 1)
    return qc


def ansatz_d(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.h(1)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    return qc


def ansatz_e(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.u(theta, math.pi / 4.0, math.pi / 4.0, 0)
    qc.cx(0, 1)
    return qc


def ansatz_f(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.x(2)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    return qc


def ansatz_g(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.h(0)
    qc.cx(0, 1)
    qc.ry(theta, 2)
    qc.cx(2, 1)
    return qc


def ansatz_h(params: tuple[float, ...]) -> QuantumCircuit:
    theta, phi = params
    qc = circuit()
    qc.ry(theta, 0)
    qc.ry(phi, 2)
    qc.cx(0, 1)
    qc.cx(2, 1)
    return qc


def ansatz_i(params: tuple[float, ...]) -> QuantumCircuit:
    theta, phi = params
    qc = circuit()
    qc.h(0)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.rx(phi, 2)
    qc.cx(2, 0)
    return qc


def ansatz_j(params: tuple[float, ...]) -> QuantumCircuit:
    theta, phi = params
    qc = circuit()
    qc.u(theta, 0.0, 0.0, 0)
    qc.cx(2, 1)
    qc.u(phi, 0.0, 0.0, 2)
    qc.cx(0, 1)
    return qc


def ansatz_k(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.x(0)
    qc.ry(theta, 0)
    qc.cx(0, 1)
    return qc


def ansatz_l(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.ry(theta, 1)
    qc.cx(1, 0)
    return qc


def ansatz_m(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.x(1)
    qc.ry(theta, 1)
    qc.cx(1, 0)
    return qc


def ansatz_n(params: tuple[float, ...]) -> QuantumCircuit:
    theta = params[0]
    qc = circuit()
    qc.u(theta, math.pi / 2.0, 0.0, 0)
    qc.cx(0, 1)
    return qc


def ansatz_o(params: tuple[float, ...]) -> QuantumCircuit:
    theta, phi = params
    qc = circuit()
    qc.ry(theta, 0)
    qc.cx(0, 1)
    qc.ry(phi, 2)
    qc.cx(2, 1)
    qc.cx(2, 0)
    return qc


ANSATZE = (
    Ansatz("A", "Ry(theta,0), CX(0,1)", ("theta",), ansatz_a),
    Ansatz("B", "H(0), Rz(theta,0), CX(0,1)", ("theta",), ansatz_b),
    Ansatz("C", "X(1), Rx(theta,0), CX(0,1)", ("theta",), ansatz_c),
    Ansatz("D", "H(1), Ry(theta,0), CX(0,1)", ("theta",), ansatz_d),
    Ansatz("E", "U(theta,pi/4,pi/4,0), CX(0,1)", ("theta",), ansatz_e),
    Ansatz("F", "X(2), Ry(theta,0), CX(0,1), CX(0,2)", ("theta",), ansatz_f),
    Ansatz("G", "H(0), CX(0,1), Ry(theta,2), CX(2,1)", ("theta",), ansatz_g),
    Ansatz("H", "Ry(theta,0), Ry(phi,2), CX(0,1), CX(2,1)", ("theta", "phi"), ansatz_h),
    Ansatz("I", "H(0), Ry(theta,0), CX(0,1), Rx(phi,2), CX(2,0)", ("theta", "phi"), ansatz_i),
    Ansatz("J", "U(theta,0,0,0), CX(2,1), U(phi,0,0,2), CX(0,1)", ("theta", "phi"), ansatz_j),
    Ansatz("K", "X(0), Ry(theta,0), CX(0,1)", ("theta",), ansatz_k),
    Ansatz("L", "Ry(theta,1), CX(1,0)", ("theta",), ansatz_l),
    Ansatz("M", "X(1), Ry(theta,1), CX(1,0)", ("theta",), ansatz_m),
    Ansatz("N", "U(theta,pi/2,0,0), CX(0,1)", ("theta",), ansatz_n),
    Ansatz("O", "Ry(theta,0), CX(0,1), Ry(phi,2), CX(2,1), CX(2,0)", ("theta", "phi"), ansatz_o),
)


def exact_expectations(ansatz: Ansatz, params: tuple[float, ...]) -> Expectations:
    state = Statevector.from_instruction(ansatz.builder(params))
    zzi = state.expectation_value(SparsePauliOp.from_list([("ZZI", 1.0)])).real
    ziz = state.expectation_value(SparsePauliOp.from_list([("ZIZ", 1.0)])).real
    ixx = state.expectation_value(SparsePauliOp.from_list([("IXX", 1.0)])).real
    return Expectations(float(zzi), float(ziz), float(ixx))


def exact_energy(ansatz: Ansatz, params: tuple[float, ...]) -> float:
    wrapped_params = tuple(float(value % TWO_PI) for value in params)
    return exact_expectations(ansatz, wrapped_params).energy


def grid_seeds(parameter_count: int, grid_points: int) -> list[tuple[float, ...]]:
    values = np.linspace(0.0, TWO_PI, grid_points, endpoint=False)
    if parameter_count == 1:
        return [(float(theta),) for theta in values]
    return [(float(theta), float(phi)) for theta in values for phi in values]


def optimize_ansatz(ansatz: Ansatz, grid_points: int) -> tuple[tuple[float, ...], Expectations]:
    seeds = grid_seeds(len(ansatz.parameter_names), grid_points)
    best_seed = min(seeds, key=lambda seed: exact_energy(ansatz, seed))

    result = minimize(
        lambda values: exact_energy(ansatz, tuple(float(value) for value in values)),
        np.array(best_seed),
        method="Nelder-Mead",
        options={"maxiter": 1200, "xatol": 1e-12, "fatol": 1e-12},
    )
    params = tuple(float(value % TWO_PI) for value in result.x)
    return params, exact_expectations(ansatz, params)


def probabilities_from_counts(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    return {bitstring: counts.get(bitstring, 0) / total for bitstring in BITSTRINGS}


def expectation_from_probabilities(
    probabilities: dict[str, float], qubits: tuple[int, ...]
) -> float:
    weighted_sum = 0.0
    for bitstring, probability in probabilities.items():
        bits_by_qubit = bitstring[::-1]
        eigenvalue = 1
        for qubit in qubits:
            eigenvalue *= 1 if bits_by_qubit[qubit] == "0" else -1
        weighted_sum += eigenvalue * probability
    return weighted_sum


def run_counts(circuit_to_run: QuantumCircuit, shots: int, seed: int) -> dict[str, int]:
    measured = circuit_to_run.copy()
    measured.measure_all()
    simulator = AerSimulator()
    compiled = transpile(measured, simulator)
    result = simulator.run(compiled, shots=shots, seed_simulator=seed).result()
    return result.get_counts(compiled)


def sampled_energy(ansatz: Ansatz, params: tuple[float, ...], shots: int, seed: int) -> float:
    zzi_probs = probabilities_from_counts(run_counts(ansatz.builder(params), shots, seed))
    ziz_probs = probabilities_from_counts(run_counts(ansatz.builder(params), shots, seed + 1))

    ixx_circuit = ansatz.builder(params)
    ixx_circuit.h(0)
    ixx_circuit.h(1)
    ixx_probs = probabilities_from_counts(run_counts(ixx_circuit, shots, seed + 2))

    zzi = expectation_from_probabilities(zzi_probs, (2, 1))
    ziz = expectation_from_probabilities(ziz_probs, (2, 0))
    ixx = expectation_from_probabilities(ixx_probs, (1, 0))
    return Expectations(zzi=zzi, ziz=ziz, ixx=ixx).energy


def rank_ansatze(args: argparse.Namespace) -> list[RankingResult]:
    results = []
    for ansatz_index, ansatz in enumerate(ANSATZE):
        params, expectations = optimize_ansatz(ansatz, args.grid_points)
        sampled = tuple(
            sampled_energy(
                ansatz,
                params,
                args.shots,
                args.seed + 100 * ansatz_index + 3 * run_index,
            )
            for run_index in range(args.runs)
        )
        results.append(
            RankingResult(
                ansatz=ansatz,
                parameters=params,
                exact_expectations=expectations,
                sampled_energies=sampled,
            )
        )
    return sorted(results, key=lambda result: result.exact_energy)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def format_parameters(result: RankingResult) -> str:
    return ", ".join(
        f"{name}={value:.6f}"
        for name, value in zip(result.ansatz.parameter_names, result.parameters, strict=True)
    )


def render_table(headers: list[str], rows: list[list[str]]) -> None:
    widths = [
        max(len(str(row[column])) for row in [headers, *rows])
        for column in range(len(headers))
    ]

    def render(row: list[str]) -> str:
        return "| " + " | ".join(
            str(value).ljust(widths[index]) for index, value in enumerate(row)
        ) + " |"

    separator = "|-" + "-|-".join("-" * width for width in widths) + "-|"
    print(render(headers))
    print(separator)
    for row in rows:
        print(render(row))


def print_ranking_table(results: list[RankingResult]) -> None:
    headers = [
        "Rank",
        "Ansatz",
        "Circuit",
        "Optimal parameters",
        "Exact E",
        "Sim mean E",
        "Var",
        "sigma",
    ]
    rows = []
    for rank, result in enumerate(results, start=1):
        rows.append(
            [
                str(rank),
                result.ansatz.label,
                result.ansatz.description,
                format_parameters(result),
                format_float(result.exact_energy),
                format_float(result.sampled_mean),
                format_float(result.sampled_variance),
                format_float(result.sampled_sigma),
            ]
        )

    print("Minimization of E -- Ansatz Determination Table")
    render_table(headers, rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rank homework ansaetze A-O.")
    parser.add_argument("--shots", type=int, default=1024, help="Shots per operator sample.")
    parser.add_argument("--runs", type=int, default=5, help="Sampled runs per ansatz.")
    parser.add_argument("--grid-points", type=int, default=41, help="Coarse optimizer grid size.")
    parser.add_argument("--seed", type=int, default=20260515, help="Base simulator seed.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = rank_ansatze(args)
    print_ranking_table(results)


if __name__ == "__main__":
    main()
