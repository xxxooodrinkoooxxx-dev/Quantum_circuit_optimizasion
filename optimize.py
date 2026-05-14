import random
import copy

from pathlib import Path
from pprint import pprint
from contextlib import redirect_stdout

from qiskit import QuantumCircuit, transpile
from qiskit import qasm2

import pyzx as zx
import pyzx.local_search.congruences as cong


# NISQ評価用の基底ゲート
NISQ_BASIS_GATES = ["rz", "sx", "x", "cx"]


# QASMファイルをQiskit回路として読み込む
def load_qasm_circuit(qasm_path):
  qc = QuantumCircuit.from_qasm_file(str(qasm_path))
  return qc


# QASMBenchから指定した回路を読み込む
def load_qasmbench_circuits(qasm_dir, max_files):
  qasm_dir = Path(qasm_dir)
  circuits = []

  paths = [
    # *qasm_dir.rglob("adder_n28_transpiled.qasm"),
    # *qasm_dir.rglob("wstate_n3_transpiled.qasm"),
    # *qasm_dir.rglob("qft_n4_transpiled.qasm"),
    # *qasm_dir.rglob("qaoa_n6_transpiled.qasm"),
    # *qasm_dir.rglob("adder_n10_transpiled.qasm"),
    *qasm_dir.rglob("multiplier_n45_transpiled.qasm")
  ]

  for path in paths:
    circuits.append((path.stem, load_qasm_circuit(path)))

    if len(circuits) >= max_files:
      break

  return circuits


# Mybench内の自作QASM回路を読み込む
def load_mybench_circuits(mybench_dir, max_files=None):
  mybench_dir = Path(mybench_dir)
  circuits = []

  qasm_paths = sorted(mybench_dir.rglob("*.qasm"))

  for path in qasm_paths:
    qc = load_qasm_circuit(path)
    circuits.append((path.stem, qc))

    if max_files is not None and len(circuits) >= max_files:
      break

  return circuits


# 回路名から回路タイプを取得
def get_circuit_type(name):
  if name.startswith("t_tail"):
    return "t_tail"

  if name.startswith("t_distributed"):
    return "t_distributed"

  if name.startswith("cx_heavy"):
    return "cx_heavy"

  if name.startswith("t_focused"):
    return "t_focused"

  if "multiplier" in name:
    return "qasmbench_multiplier"

  return "unknown"


# 回路タイプに応じてLC/Pivotの強さを決める
def get_lc_pivot_params(name, qc):
  circuit_type = get_circuit_type(name)

  if qc.num_qubits >= 30:
    return 10, 5

  if circuit_type == "cx_heavy":
    return 50, 15

  if circuit_type == "t_distributed":
    return 40, 15

  if circuit_type == "t_tail":
    return 30, 10

  if circuit_type == "t_focused":
    return 20, 10

  return 20, 10


# 回路の指標を取得
def get_metrics(qc):
  ops = qc.count_ops()

  one_qubit_count = get_one_qubit_count(qc)
  two_qubit_count = get_two_qubit_count(qc)

  explicit_two_qubit_count = (
    ops.get("cx", 0)
    + ops.get("cz", 0)
    + ops.get("swap", 0)
  )

  metrics = {
    "num_qubits": qc.num_qubits,
    "depth": qc.depth(),
    "size": qc.size(),

    "one_qubit_count": one_qubit_count,
    "two_qubit_count": two_qubit_count,
    "explicit_two_qubit_count": explicit_two_qubit_count,

    "h_count": ops.get("h", 0),
    "x_count": ops.get("x", 0),
    "y_count": ops.get("y", 0),
    "z_count": ops.get("z", 0),
    "s_count": ops.get("s", 0),
    "sdg_count": ops.get("sdg", 0),

    "t_count": ops.get("t", 0) + ops.get("tdg", 0),
    "t_gate_count": ops.get("t", 0),
    "tdg_gate_count": ops.get("tdg", 0),

    "rx_count": ops.get("rx", 0),
    "ry_count": ops.get("ry", 0),
    "rz_count": ops.get("rz", 0),
    "sx_count": ops.get("sx", 0),

    "u1_count": ops.get("u1", 0),
    "u2_count": ops.get("u2", 0),
    "u3_count": ops.get("u3", 0),
    "unitary_count": ops.get("unitary", 0),

    "cx_count": ops.get("cx", 0),
    "cz_count": ops.get("cz", 0),
    "swap_count": ops.get("swap", 0),

    "measure_count": ops.get("measure", 0),
    "barrier_count": ops.get("barrier", 0)
  }

  return metrics


# 1量子ビットゲートをカウント
def get_one_qubit_count(qc):
  count = 0

  for instruction in qc.data:
    if instruction.operation.num_qubits == 1:
      count += 1

  return count


# 2量子ビットゲートをカウント
def get_two_qubit_count(qc):
  count = 0

  for instruction in qc.data:
    if instruction.operation.num_qubits == 2:
      count += 1

  return count


# measurement qubitとclassical bitの対応を取得
def get_measure_map(qc):
  measure_map = []

  for instruction in qc.data:
    inst = instruction.operation
    qargs = instruction.qubits
    cargs = instruction.clbits

    if inst.name == "measure":
      q_index = qc.find_bit(qargs[0]).index
      c_index = qc.find_bit(cargs[0]).index
      measure_map.append((q_index, c_index))

  return measure_map


# PyZXで扱うためにmesurementを外す
def remove_measurements_for_pyzx(qc):
  measure_map = get_measure_map(qc)
  unitary_qc = qc.remove_final_measurements(inplace=False)

  return unitary_qc, measure_map


# PyZX最適化後の回路にmesurementを戻す
def restore_measurements(qc, measure_map):
  if not measure_map:
    return qc

  num_clbits = max(c_index for _, c_index in measure_map) + 1

  final_qc = QuantumCircuit(qc.num_qubits, num_clbits)
  final_qc.compose(qc, inplace=True)

  for q_index, c_index in measure_map:
    final_qc.measure(q_index, c_index)

  return final_qc


# Qiskit回路をPyZXのZXグラフへ変換
def qiskit_to_zx_graph(qc):
  qasm_str = qasm2.dumps(qc)
  zx_circuit = zx.Circuit.from_qasm(qasm_str)
  graph = zx_circuit.to_graph()

  return graph


# ZXグラフをQiskit回路に戻し、指定basisで再transpileする
def graph_to_qiskit_circuit(graph, measure_map=None, basis_gates=None):
  if basis_gates is None:
    basis_gates = NISQ_BASIS_GATES

  optimized_zx_circuit = zx.extract_circuit(graph)
  optimized_qasm = optimized_zx_circuit.to_qasm()
  optimized_qc = QuantumCircuit.from_qasm_str(optimized_qasm)

  optimized_qc = transpile(
    optimized_qc,
    basis_gates=basis_gates,
    optimization_level=3
  )

  optimized_qc = restore_measurements(optimized_qc, measure_map)

  return optimized_qc


# NISQ向けにCX数、depth、sizeの順で評価
def score_circuit(qc):
  metrics = get_metrics(qc)

  return (
    metrics["cx_count"],
    metrics["depth"],
    metrics["size"]
  )


# FTQC向けにT-countを加えて評価
def ftqc_score_circuit(qc):
  metrics = get_metrics(qc)

  return (
    metrics["t_count"],
    metrics["cx_count"],
    metrics["depth"],
    metrics["size"]
  )


# PyZX full_reduceによる通常のZX最適化
def optimize(qc, basis_gates=None):
  if basis_gates is None:
    basis_gates = NISQ_BASIS_GATES

  unitary_qc, measure_map = remove_measurements_for_pyzx(qc)
  graph = qiskit_to_zx_graph(unitary_qc)

  zx.simplify.full_reduce(graph)

  optimized_qc = graph_to_qiskit_circuit(
    graph,
    measure_map=measure_map,
    basis_gates=basis_gates
  )

  return optimized_qc


# ランダムにLCまたはPivotをZXグラフへ適用
def apply_random_lc_or_pivot(graph, rng):
  use_lc = rng.random() < 0.5

  if use_lc:
    try:
      cong.apply_rand_lc(graph)
      return True, "lc"
    except Exception:
      return False, "lc_failed"

  try:
    cong.apply_rand_pivot(graph)
    return True, "pivot"
  except Exception:
    return False, "pivot_failed"


# LC/Pivot探索でCX数・depth・sizeが良い回路を探す
def optimize_with_lc_pivot_search(qc, num_trials=20, num_steps=10, seed=0, basis_gates=None, verbose=True):
  if basis_gates is None:
    basis_gates = NISQ_BASIS_GATES

  rng = random.Random(seed)

  unitary_qc, measure_map = remove_measurements_for_pyzx(qc)
  base_graph = qiskit_to_zx_graph(unitary_qc)

  zx.simplify.full_reduce(base_graph)

  best_qc = graph_to_qiskit_circuit(
    copy.deepcopy(base_graph),
    measure_map=measure_map,
    basis_gates=basis_gates
  )

  best_score = score_circuit(best_qc)

  stats = {
    "lc_success": 0,
    "pivot_success": 0,
    "lc_fail": 0,
    "pivot_fail": 0,
    "no_congruences_module": 0,
    "no_apply_rand_lc": 0,
    "no_apply_rand_pivot": 0,
    "extract_fail": 0,
    "trial_improved": 0
  }

  for _ in range(num_trials):
    trial_graph = copy.deepcopy(base_graph)

    for _ in range(num_steps):
      success, kind = apply_random_lc_or_pivot(trial_graph, rng)

      if success and kind == "lc":
        stats["lc_success"] += 1
      elif success and kind == "pivot":
        stats["pivot_success"] += 1
      elif kind == "lc_failed":
        stats["lc_fail"] += 1
      elif kind == "pivot_failed":
        stats["pivot_fail"] += 1
      elif kind in stats:
        stats[kind] += 1

    try:
      zx.simplify.full_reduce(trial_graph)
    except Exception:
      pass

    try:
      trial_qc = graph_to_qiskit_circuit(
        trial_graph,
        measure_map=measure_map,
        basis_gates=basis_gates
      )
    except Exception:
      stats["extract_fail"] += 1
      continue

    trial_score = score_circuit(trial_qc)

    if trial_score < best_score:
      best_score = trial_score
      best_qc = trial_qc
      stats["trial_improved"] += 1

  if verbose:
    print("LC/Pivot search stats:")
    pprint(stats)
    print("best_score:")
    pprint(best_score)
    print()

  return best_qc


# PyZXによるCX増加とLC/Pivotの回収率を計算
def get_recovery_metrics(baseline_metrics, pyzx_metrics, lc_pivot_metrics):
  pyzx_overhead = pyzx_metrics["cx_count"] - baseline_metrics["cx_count"]
  recovered_overhead = pyzx_metrics["cx_count"] - lc_pivot_metrics["cx_count"]

  if pyzx_overhead <= 0:
    recovery_rate = 0.0
  else:
    recovery_rate = recovered_overhead / pyzx_overhead

  return {
    "pyzx_overhead": pyzx_overhead,
    "recovered_overhead": recovered_overhead,
    "recovery_rate": recovery_rate
  }


# 各最適化手法を実行して比較
def main():

  output_dir = Path("results")
  output_dir.mkdir(exist_ok=True)

  output_path = output_dir / "experiment_result.txt"

  with open(output_path, "w") as f:
    with redirect_stdout(f):

      # QASMBench
      # circuits = load_qasmbench_circuits("QASMBench", max_files=1)

      # Mybench
      circuits = load_mybench_circuits("Mybench", max_files=None)

      for name, qc in circuits:

        circuit_type = get_circuit_type(name)
        num_trials, num_steps = get_lc_pivot_params(name, qc)

        print("============================================================")
        print(name)
        print("circuit_type:")
        pprint(circuit_type)
        print("lc/pivot params:")
        pprint({
          "num_trials": num_trials,
          "num_steps": num_steps
        })
        print()

        metrics = get_metrics(qc)

        baseline_qc = transpile(qc, optimization_level=3)
        baseline_metrics = get_metrics(baseline_qc)

        pyzx_qc = optimize(qc, basis_gates=NISQ_BASIS_GATES)
        pyzx_metrics = get_metrics(pyzx_qc)

        my_optimized_qc_1 = optimize(
          baseline_qc,
          basis_gates=NISQ_BASIS_GATES
        )

        my_optimized_metrics_1 = get_metrics(my_optimized_qc_1)

        my_optimized_qc_2 = transpile(
          my_optimized_qc_1,
          basis_gates=NISQ_BASIS_GATES,
          optimization_level=3
        )

        my_optimized_metrics_2 = get_metrics(my_optimized_qc_2)

        my_optimized_qc_3 = optimize_with_lc_pivot_search(
          baseline_qc,
          num_trials=num_trials,
          num_steps=num_steps,
          seed=0,
          basis_gates=NISQ_BASIS_GATES,
          verbose=True
        )

        my_optimized_metrics_3 = get_metrics(my_optimized_qc_3)

        recovery_metrics = get_recovery_metrics(
          baseline_metrics,
          my_optimized_metrics_1,
          my_optimized_metrics_3
        )

        print("original:")
        pprint(metrics)
        print()

        print("qiskit baseline:")
        pprint(baseline_metrics)
        print()

        print("pyzx:")
        pprint(pyzx_metrics)
        print()

        print("qiskit + pyzx:")
        pprint(my_optimized_metrics_1)
        print()

        print("qiskit + pyzx + qiskit:")
        pprint(my_optimized_metrics_2)
        print()

        print("qiskit + pyzx + lc/pivot:")
        pprint(my_optimized_metrics_3)
        print()

        print("recovery metrics:")
        pprint(recovery_metrics)
        print()


if __name__ == "__main__":
  main()