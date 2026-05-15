"""Run a simple Bell-state quantum circuit on a local Qiskit Aer simulator."""

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator


def build_bell_circuit() -> QuantumCircuit:
    """Create a two-qubit Bell-state circuit with measurements."""
    circuit = QuantumCircuit(2, 2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.measure([0, 1], [0, 1])
    return circuit


def main() -> None:
    circuit = build_bell_circuit()
    simulator = AerSimulator()
    compiled_circuit = transpile(circuit, simulator)
    job = simulator.run(compiled_circuit, shots=1024)
    result = job.result()

    print("Bell-state circuit:")
    print(circuit.draw(output="text"))
    print("Simulation counts:")
    print(result.get_counts(compiled_circuit))


if __name__ == "__main__":
    main()
