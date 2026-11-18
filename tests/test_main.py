"""Tests for QuantumMachineLearning-Lab: simulator, dataset, variational
classifier (exact parameter-shift gradients), quantum kernel, classical
baselines, measured barren plateaus, run-diagnosis classifier, gates."""
import math

import pytest

import main as m


# ---- statevector simulator
def test_zero_state_norm():
    assert abs(m.state_norm(m.zero_state(3)) - 1.0) < 1e-12


def test_ry_pi_flips():
    out = m.ry(m.zero_state(1), 1, 0, math.pi)
    assert abs(abs(out[1]) - 1.0) < 1e-12


def test_hadamard_superposition():
    out = m.hadamard(m.zero_state(1), 1, 0)
    assert abs(abs(out[0]) ** 2 - 0.5) < 1e-12
    assert abs(abs(out[1]) ** 2 - 0.5) < 1e-12


def test_cnot_bell_state():
    bell = m.cnot(m.hadamard(m.zero_state(2), 2, 0), 2, 0, 1)
    assert abs(abs(bell[0]) ** 2 - 0.5) < 1e-12
    assert abs(bell[1]) < 1e-12 and abs(bell[2]) < 1e-12
    assert abs(abs(bell[3]) ** 2 - 0.5) < 1e-12


def test_expectation_z():
    assert abs(m.expectation_z(m.zero_state(1), 1, 0) - 1.0) < 1e-12
    flipped = m.ry(m.zero_state(1), 1, 0, math.pi)
    assert abs(m.expectation_z(flipped, 1, 0) + 1.0) < 1e-12


def test_parity_expectations():
    assert abs(m.expectation_parity(m.zero_state(3), 3) - 1.0) < 1e-12
    odd = m.ry(m.zero_state(2), 2, 0, math.pi)
    assert abs(m.expectation_parity(odd, 2) + 1.0) < 1e-12


# ---- dataset
def test_dataset_sizes():
    assert len(m.TRAIN) == 40 and len(m.TEST) == 20


def test_dataset_deterministic():
    train2, test2 = m.make_dataset()
    assert train2 == m.TRAIN and test2 == m.TEST


def test_dataset_balanced():
    all_points = m.TRAIN + m.TEST
    assert sum(1 for _, c in all_points if c == 0) == m.POINTS_PER_CLASS
    assert sum(1 for _, c in all_points if c == 1) == m.POINTS_PER_CLASS


def test_dataset_report_shape():
    rep = m.dataset_report()
    assert rep["total"] == 60 and rep["train"] == 40 and rep["test"] == 20


# ---- variational classifier
def test_parameter_shift_matches_finite_difference():
    theta = m._initial_theta()
    grads = m.variational_gradient(theta)
    eps = 1e-5
    for i in (0, 5, 15):
        plus = list(theta)
        plus[i] += eps
        minus = list(theta)
        minus[i] -= eps
        fd = (m.variational_loss(plus) - m.variational_loss(minus)) / (2 * eps)
        assert abs(fd - grads[i]) < 1e-6


def test_variational_training_improves():
    run = m.variational_run()
    assert run["final_loss"] < 0.5 * run["init_loss"]
    assert run["test_accuracy"] > run["init_test_accuracy"]


def test_variational_accuracy_bars():
    run = m.variational_run()
    assert run["test_accuracy"] >= 0.85
    assert run["train_accuracy"] >= 0.85


def test_variational_param_count():
    assert m.N_PARAMS == 16
    assert len(m.variational_run()["theta"]) == 16


def test_variational_deterministic_short_run():
    a = m.train_variational(epochs=3, lr=0.25)
    b = m.train_variational(epochs=3, lr=0.25)
    assert a["final_loss"] == b["final_loss"]
    assert a["theta"] == b["theta"]


def test_variational_circuit_normalized():
    state = m.variational_circuit(m.TRAIN[0][0], m._initial_theta())
    assert abs(m.state_norm(state) - 1.0) < 1e-9


# ---- quantum kernel
def test_kernel_self_is_one():
    x = m.TRAIN[0][0]
    assert abs(m.quantum_kernel(x, x) - 1.0) < 1e-9


def test_kernel_symmetric_and_bounded():
    a, b = m.TRAIN[0][0], m.TRAIN[1][0]
    assert abs(m.quantum_kernel(a, b) - m.quantum_kernel(b, a)) < 1e-12
    assert 0.0 <= m.quantum_kernel(a, b) <= 1.0 + 1e-12


def test_kernel_accuracy_bars():
    rep = m.kernel_report()
    assert rep["test_accuracy"] >= 0.90
    assert rep["train_accuracy"] >= 0.90


# ---- classical baselines
def test_logistic_regression_struggles():
    acc = m.baseline_report()["logistic_regression"]["test_accuracy"]
    assert 0.40 <= acc <= 0.75


def test_rbf_gamma_chosen_by_cv():
    rbf = m.baseline_report()["rbf_centroid"]
    assert rbf["gamma"] in m.RBF_GAMMA_GRID
    assert rbf["loo_scores"][str(rbf["gamma"])] == max(
        rbf["loo_scores"].values())


def test_rbf_accuracy_bar():
    assert m.baseline_report()["rbf_centroid"]["test_accuracy"] >= 0.90


def test_classical_baseline_competitive():
    head = m.headline_report()
    assert head["classical_rbf_centroid"] >= head["variational_circuit"]
    assert head["quantum_advantage_claimed"] is False


def test_headline_all_models_present():
    head = m.headline_report()
    for key in ("variational_circuit", "quantum_kernel",
                "classical_rbf_centroid", "classical_logistic_regression"):
        assert 0.0 <= head[key] <= 1.0


# ---- barren plateaus
def test_plateau_variance_decays():
    var = m.plateau_report()["gradient_variance"]
    assert all(v > 0 for v in var.values())
    assert var["6"] < var["2"] / 4


def test_plateau_fit_slope_negative():
    rep = m.plateau_report()
    assert rep["log10_slope_per_qubit"] < 0
    assert rep["decay_factor_2_to_6_qubits"] > 4


# ---- run-diagnosis classifier
def test_diagnosis_scenarios():
    diag = m.diagnosis_report()
    assert diag["variational_run_100_epochs"] == "converged"
    assert diag["stalled_run_5_epochs_tiny_lr"] == "plateaued"
    assert diag["six_qubit_random_circuit"] == "barren"


def test_diagnosis_matches_policy():
    for features in [(0.8, 0.05, 0.95), (0.05, 0.05, 0.55),
                     (0.0, 0.003, 0.5), (0.9, 0.2, 0.95),
                     (0.25, 0.05, 0.7)]:
        assert m.classify_run(*features) == m.diagnose_run(*features)


def test_diagnosis_policy_thresholds():
    assert m.diagnose_run(0.9, 0.005, 0.99) == "barren"
    assert m.diagnose_run(0.9, 0.05, 0.9) == "converged"
    assert m.diagnose_run(0.2, 0.05, 0.9) == "plateaued"


# ---- HOLD gates + council seats
def test_hold_gates_14():
    assert len(m.HOLD_GATES) == 14
    assert all(v.startswith("HOLD") for v in m.HOLD_GATES.values())


@pytest.mark.parametrize("action", [
    "quantum_advantage_claim", "production_ml_system_claim",
    "benchmark_superiority_claim", "hardware_execution_claim",
    "medical_diagnosis_claim", "financial_prediction_claim",
])
def test_gated_actions_hold(action):
    assert m.gate(action) == "HOLD"


def test_sandbox_proceeds():
    assert m.gate("run_simulation") == "PROCEED"


def test_hold_demo_exit_code():
    assert m.main(["hold-demo", "quantum_advantage_claim"]) == 2
    assert m.main(["hold-demo", "run_simulation"]) == 0


def test_seats_cannot_override():
    brains = m.brains_report()
    assert brains["decider"]["can_override_hold"] is False
    assert brains["decider"]["seat"] == "Gemini Argon 4 Extended Thinking"
    assert brains["advisor"]["can_override_hold"] is False
    assert brains["flow"]["rendered"] is False
    assert brains["flow"]["live"] == 0


def test_checks_all_pass():
    checks = m.run_checks()
    assert all(ok for _, ok in checks)
    assert len(checks) == 48
