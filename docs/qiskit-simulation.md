# Qiskit local simulation setup

This repository includes a small Python environment for running quantum
computing circuits on a local simulator from Cursor.

## Create and activate the environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-qiskit.txt
```

## Run the included Bell-state simulation

```bash
python examples/qiskit_bell_simulation.py
```

The script builds a two-qubit Bell circuit, runs it on Qiskit Aer, and prints
the simulated measurement counts. A healthy run should produce mostly `00` and
`11` results.

## Add your own circuits

Create another Python script under `examples/`, import Qiskit, and run it with
the same activated `.venv`:

```python
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

circuit = QuantumCircuit(1, 1)
circuit.h(0)
circuit.measure(0, 0)

simulator = AerSimulator()
compiled = transpile(circuit, simulator)
result = simulator.run(compiled, shots=1024).result()
print(result.get_counts(compiled))
```
