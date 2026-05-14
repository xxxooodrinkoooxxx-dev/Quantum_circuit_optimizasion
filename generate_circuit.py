import random

from pathlib import Path

from qiskit import qasm2
from qiskit.quantum_info.random import random_clifford


def generate_random_clifford(num_qubits, seed=None):
  clifford = random_clifford(num_qubits, seed=seed)

  qc = clifford.to_circuit()

  return qc


def add_random_t_gates(qc, num_t_gates, seed=None):
  rng = random.Random(seed)

  for _ in range(num_t_gates):

    qubit = rng.randrange(qc.num_qubits)

    if rng.random() < 0.5:
      qc.t(qubit)
    else:
      qc.tdg(qubit)

  return qc


def generate_clifford_t_circuit(num_qubits, num_t_gates, seed=None):

  qc = generate_random_clifford(num_qubits=num_qubits, seed=seed)

  qc = add_random_t_gates(
    qc=qc,
    num_t_gates=num_t_gates,
    seed=seed
  )

  return qc


def save_circuit_qasm(qc, save_dir, filename):
  save_dir = Path(save_dir)

  save_dir.mkdir(parents=True, exist_ok=True)

  qasm_str = qasm2.dumps(qc)

  save_path = save_dir / f"{filename}.qasm"

  with open(save_path, "w") as f:
    f.write(qasm_str)

  return save_path


def main():

  save_dir = "Mybench"

  for i in range(10):

    qc = generate_clifford_t_circuit(
      num_qubits=8,
      num_t_gates=20,
      seed=i
    )

    filename = f"clifford_t_n8_t20_seed{i}"

    save_path = save_circuit_qasm(
      qc=qc,
      save_dir=save_dir,
      filename=filename
    )

    print(save_path)


if __name__ == "__main__":
  main()