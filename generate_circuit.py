import random

from pathlib import Path

from qiskit import QuantumCircuit
from qiskit import qasm2
from qiskit.quantum_info.random import random_clifford


# random Clifford回路を生成
def generate_random_clifford(num_qubits, seed=None):
  clifford = random_clifford(num_qubits, seed=seed)
  qc = clifford.to_circuit()

  return qc


# ランダムなT/Tdgゲートを追加
def add_random_t_gates(qc, num_t_gates, seed=None):
  rng = random.Random(seed)

  for _ in range(num_t_gates):
    qubit = rng.randrange(qc.num_qubits)

    if rng.random() < 0.5:
      qc.t(qubit)
    else:
      qc.tdg(qubit)

  return qc


# ランダムなCXゲートを追加
def add_random_cx_gates(qc, num_cx_gates, seed=None):
  rng = random.Random(seed)

  for _ in range(num_cx_gates):
    control = rng.randrange(qc.num_qubits)
    target = rng.randrange(qc.num_qubits)

    while target == control:
      target = rng.randrange(qc.num_qubits)

    qc.cx(control, target)

  return qc


# Tゲートを末尾にまとめたClifford+T回路を生成
def generate_t_tail_circuit(num_qubits, num_t_gates, seed=None):
  qc = generate_random_clifford(num_qubits, seed=seed)
  qc = add_random_t_gates(qc, num_t_gates, seed=seed)

  return qc


# Clifford層とT層を交互に入れたClifford+T回路を生成
def generate_t_distributed_circuit(num_qubits, num_layers, t_per_layer, seed=None):
  qc = QuantumCircuit(num_qubits)
  rng = random.Random(seed)

  for layer in range(num_layers):
    clifford_seed = rng.randrange(10**9)
    t_seed = rng.randrange(10**9)

    layer_qc = generate_random_clifford(num_qubits, seed=clifford_seed)
    qc.compose(layer_qc, inplace=True)
    qc = add_random_t_gates(qc, t_per_layer, seed=t_seed)

  return qc


# CXを多めに含むClifford+T回路を生成
def generate_cx_heavy_circuit(num_qubits, num_layers, cx_per_layer, t_per_layer, seed=None):
  qc = QuantumCircuit(num_qubits)
  rng = random.Random(seed)

  for layer in range(num_layers):
    clifford_seed = rng.randrange(10**9)
    cx_seed = rng.randrange(10**9)
    t_seed = rng.randrange(10**9)

    layer_qc = generate_random_clifford(num_qubits, seed=clifford_seed)
    qc.compose(layer_qc, inplace=True)
    qc = add_random_cx_gates(qc, cx_per_layer, seed=cx_seed)
    qc = add_random_t_gates(qc, t_per_layer, seed=t_seed)

  return qc


# CXを少なめにしてT配置の影響を見やすくした回路を生成
def generate_t_focused_circuit(num_qubits, num_layers, t_per_layer, seed=None):
  qc = QuantumCircuit(num_qubits)
  rng = random.Random(seed)

  for layer in range(num_layers):
    t_seed = rng.randrange(10**9)

    for qubit in range(num_qubits):
      if rng.random() < 0.5:
        qc.h(qubit)
      else:
        qc.s(qubit)

    qc = add_random_t_gates(qc, t_per_layer, seed=t_seed)

  return qc


# QASMとして保存
def save_circuit_qasm(qc, save_dir, filename):
  save_dir = Path(save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)

  qasm_str = qasm2.dumps(qc)
  save_path = save_dir / f"{filename}.qasm"

  with open(save_path, "w") as f:
    f.write(qasm_str)

  return save_path


# 生成した回路を保存
def generate_and_save(qc, save_dir, filename):
  save_path = save_circuit_qasm(qc, save_dir, filename)
  print(save_path)


def main():
  save_dir = "Mybench"
  num_seeds = 10

  for seed in range(num_seeds):
    qc = generate_t_tail_circuit(num_qubits=8, num_t_gates=20, seed=seed)
    generate_and_save(qc, save_dir, f"t_tail_n8_t20_seed{seed}")

    qc = generate_t_distributed_circuit(num_qubits=8, num_layers=4, t_per_layer=5, seed=seed)
    generate_and_save(qc, save_dir, f"t_distributed_n8_l4_t20_seed{seed}")

    qc = generate_cx_heavy_circuit(num_qubits=8, num_layers=4, cx_per_layer=10, t_per_layer=5, seed=seed)
    generate_and_save(qc, save_dir, f"cx_heavy_n8_l4_cx40_t20_seed{seed}")

    qc = generate_t_focused_circuit(num_qubits=8, num_layers=4, t_per_layer=5, seed=seed)
    generate_and_save(qc, save_dir, f"t_focused_n8_l4_t20_seed{seed}")


if __name__ == "__main__":
  main()