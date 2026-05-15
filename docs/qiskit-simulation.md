# Qiskit local simulation setup

This repository includes a small Python environment for running quantum
computing circuits on a local simulator from Cursor.

## Create and activate the environment

If `python3 -m venv .venv` reports that `ensurepip` is unavailable on
Ubuntu/Debian, install venv support first:

```bash
sudo apt-get update
sudo apt-get install -y python3.12-venv
```

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

## Run the homework Ansatz O simulation

```bash
python examples/qiskit_ansatz_o_vqe.py
```

This script implements Appendix 1 Ansatz O:

```text
Ry(theta, q0), CX(0,1), Ry(phi, q2), CX(2,1), CX(2,0)
```

It optimizes `theta` and `phi` against
`H = 2 * ZZI + ZIZ - IXX`, then estimates the energy with five 1024-shot Aer
simulator runs. To evaluate your own parameters instead of optimizing:

```bash
python examples/qiskit_ansatz_o_vqe.py --theta 2.82 --phi 1.87
```

## Generate the all-ansatz determination table

```bash
python examples/qiskit_ansatz_determination_table.py
```

This ranks Appendix 1 ansätze A through O by optimized exact energy for
`H = 2 * ZZI + ZIZ - IXX`. It also runs five independent 1024-shot simulator
estimates per ansatz and prints the sampled mean, variance, and standard
deviation.

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
