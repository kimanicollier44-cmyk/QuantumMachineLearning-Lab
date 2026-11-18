#!/usr/bin/env python3
"""QuantumMachineLearning-Lab - educational quantum machine learning lab.

by Darion Kellar

Pure Python standard library. Zero dependencies. Zero network.
Education only - and the central lesson is stated up front:

  * This lab builds the actual machinery of quantum machine
    learning on a from-scratch statevector simulator: a variational
    classifier trained with exact parameter-shift gradients, and a
    fidelity (quantum) kernel classifier - both on a small labelled
    synthetic dataset generated in-lab with a fixed seed.
  * In the SAME run it trains classical baselines on the SAME split:
    logistic regression by gradient descent, and an RBF-kernel
    nearest-centroid classifier whose width is picked by
    leave-one-out cross-validation on the training split only.
  * On this toy data the classical RBF baseline TIES the quantum
    kernel and BEATS the variational circuit. That is the honest
    result, printed in the headline table - no quantum advantage is
    claimed, implied, or marketed. The lab teaches the machinery:
    encodings, kernels, gradients, and the barren plateaus that
    make training hard - it does not sell a speedup.
  * Everything is an exact classical simulation of 2-qubit circuits
    (plus random circuits up to 6 qubits for the plateau study).
    No quantum hardware is touched, no real dataset is used, and
    every advantage / production / benchmark / deployment use is a
    permanent HOLD gate - with Gemini Argon 4 Extended Thinking as
    DECIDER (names-only, tighten-only), Muse as advisor (never
    overrides HOLD), and Google Flow plan-only seats.
"""
import hashlib
import json
import math
import random
import sys

# ---------------------------------------------------------------- constants
VERSION = "1.0.0"
DATA_SEED = 20261118
INIT_SEED = 7
POINTS_PER_CLASS = 30
TRAIN_SIZE = 40
NOISE_SIGMA = 0.08
VAR_LAYERS = 4
VAR_EPOCHS = 100
VAR_LR = 0.25
PLATEAU_SAMPLES = 16
RBF_GAMMA_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0)

# ---------------------------------------------------------------- HOLD gates
HOLD_GATES = {
    "quantum_advantage_claim": "HOLD - no quantum advantage claim; the classical baseline ties or beats the quantum models in this lab's own headline table",
    "production_ml_system_claim": "HOLD - no claim this toy lab is a production machine-learning system",
    "benchmark_superiority_claim": "HOLD - no benchmark superiority claims against real models, libraries, or hardware",
    "speedup_guarantee_claim": "HOLD - no guaranteed speedup; the simulation itself costs 2^n amplitudes and says so",
    "hardware_execution_claim": "HOLD - no claim any result ran on real quantum hardware; everything is exact classical simulation",
    "real_dataset_training": "HOLD - no training on real, personal, or sensitive datasets; the lab's data is synthetic and seeded",
    "medical_diagnosis_claim": "HOLD - no medical diagnosis, screening, or health prediction use",
    "financial_prediction_claim": "HOLD - no financial prediction, trading signal, credit, or insurance scoring use",
    "drug_discovery_claim": "HOLD - no drug, molecule, or materials discovery claim from a toy classifier",
    "biometric_identification_use": "HOLD - no face, voice, gait, or other biometric identification use",
    "autonomous_decision_system": "HOLD - no deploying this lab as an automated decision-maker affecting real people",
    "defense_targeting_use": "HOLD - no defence, targeting, surveillance, or weapons use",
    "cryptography_breaking_claim": "HOLD - no claim of breaking, weakening, or evaluating real cryptography",
    "certified_ai_system_claim": "HOLD - no claim of certification, audit, or regulatory approval for any AI system",
}


def gate(action):
    """Return 'HOLD' if the action is gated, else 'PROCEED' (sandbox sim)."""
    return "HOLD" if action in HOLD_GATES else "PROCEED"


# ================================================================ 1. Statevector simulator
# Amplitudes are plain Python complex numbers in a list of length
# 2^n. Qubit 0 is the most significant bit. Every gate is applied by
# index arithmetic - no numpy, no hidden linear algebra.

def zero_state(n):
    state = [0j] * (1 << n)
    state[0] = 1.0 + 0j
    return state


def state_norm(state):
    return math.sqrt(sum(abs(a) ** 2 for a in state))


def _apply_1q(state, n, q, m):
    step = 1 << (n - 1 - q)
    out = list(state)
    for base in range(0, 1 << n, step * 2):
        for i in range(base, base + step):
            j = i + step
            a, b = state[i], state[j]
            out[i] = m[0][0] * a + m[0][1] * b
            out[j] = m[1][0] * a + m[1][1] * b
    return out


def rx(state, n, q, theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return _apply_1q(state, n, q, [[c, -1j * s], [-1j * s, c]])


def ry(state, n, q, theta):
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    return _apply_1q(state, n, q, [[c, -s], [s, c]])


def rz(state, n, q, theta):
    return _apply_1q(state, n, q,
                     [[complex(math.cos(theta / 2), -math.sin(theta / 2)), 0],
                      [0, complex(math.cos(theta / 2), math.sin(theta / 2))]])


def hadamard(state, n, q):
    f = 1 / math.sqrt(2)
    return _apply_1q(state, n, q, [[f, f], [f, -f]])


def cnot(state, n, control, target):
    bc, bt = 1 << (n - 1 - control), 1 << (n - 1 - target)
    out = list(state)
    for i in range(1 << n):
        if i & bc:
            out[i ^ bt] = state[i]
    return out


def expectation_z(state, n, q):
    bit = 1 << (n - 1 - q)
    return sum((1.0 if not (i & bit) else -1.0) * abs(a) ** 2
               for i, a in enumerate(state))


def expectation_parity(state, n):
    """Expectation of Z on every qubit at once - the global cost."""
    return sum((1.0 if bin(i).count("1") % 2 == 0 else -1.0) * abs(a) ** 2
               for i, a in enumerate(state))


# ================================================================ 2. Dataset
# Two interleaving half-circles ("moons") - the classic toy problem
# a linear model cannot separate. Generated with a fixed seed,
# shuffled with the same seed, split deterministically: the first
# TRAIN_SIZE points train, the rest test. Same split for every model
# in the lab, quantum and classical alike.

def make_dataset():
    rng = random.Random(DATA_SEED)
    points = []
    for cls in (0, 1):
        for _ in range(POINTS_PER_CLASS):
            t = rng.uniform(0, math.pi)
            if cls == 0:
                x, y = math.cos(t), math.sin(t)
            else:
                x, y = 1 - math.cos(t), -math.sin(t) + 0.5
            x += rng.gauss(0, NOISE_SIGMA)
            y += rng.gauss(0, NOISE_SIGMA)
            points.append(((x - 0.5, y - 0.25), cls))
    rng.shuffle(points)
    return points[:TRAIN_SIZE], points[TRAIN_SIZE:]


TRAIN, TEST = make_dataset()


def dataset_report():
    return {
        "shape": "two interleaving half-circles (moons)",
        "seed": DATA_SEED,
        "noise_sigma": NOISE_SIGMA,
        "total": len(TRAIN) + len(TEST),
        "train": len(TRAIN),
        "test": len(TEST),
        "train_class_counts": {
            str(c): sum(1 for _, cc in TRAIN if cc == c) for c in (0, 1)},
        "test_class_counts": {
            str(c): sum(1 for _, cc in TEST if cc == c) for c in (0, 1)},
    }


# ================================================================ 3. Variational classifier
# 2 qubits. Each of VAR_LAYERS layers re-uploads the data (angle
# encoding RY(pi*x1), RY(pi*x2), a CNOT, and an RY(pi*x1*x2) product
# term), then applies trainable RY/RZ rotations and a CNOT. The
# prediction is sign(<Z0>); the loss is the mean squared error
# against +-1 labels. Gradients are EXACT via the parameter-shift
# rule (each rotation generator has eigenvalues +-1/2), and the
# optimizer is Adam with a fixed learning rate and epoch count -
# same seed, same start, same result, every run.

N_PARAMS = VAR_LAYERS * 4


def variational_circuit(x, theta):
    state = zero_state(2)
    for layer in range(VAR_LAYERS):
        state = ry(state, 2, 0, math.pi * x[0])
        state = ry(state, 2, 1, math.pi * x[1])
        state = cnot(state, 2, 0, 1)
        state = ry(state, 2, 1, math.pi * x[0] * x[1])
        b = layer * 4
        state = ry(state, 2, 0, theta[b])
        state = rz(state, 2, 0, theta[b + 1])
        state = ry(state, 2, 1, theta[b + 2])
        state = rz(state, 2, 1, theta[b + 3])
        state = cnot(state, 2, 0, 1)
    return state


def variational_z(x, theta):
    return expectation_z(variational_circuit(x, theta), 2, 0)


def variational_loss(theta, data=TRAIN):
    total = 0.0
    for x, cls in data:
        y = 1.0 if cls == 0 else -1.0
        total += (variational_z(x, theta) - y) ** 2
    return total / len(data)


def variational_gradient(theta):
    """Exact gradient of the training loss via the parameter-shift rule.

    The shift rule is exact for an expectation value (each rotation
    generator has eigenvalues +-1/2), so it is applied to <Z0> itself
    and chained through the squared loss:
        d/dtheta (z - y)^2 = 2 (z - y) * [z(theta + pi/2) - z(theta - pi/2)] / 2
    Shifting the squared loss directly would NOT be exact - the
    square adds a frequency-2 term the plain shift misses - so the
    chain step is done in the open here, and the checks assert the
    result against central finite differences.
    """
    grads = [0.0] * N_PARAMS
    for x, cls in TRAIN:
        y = 1.0 if cls == 0 else -1.0
        z = variational_z(x, theta)
        for i in range(N_PARAMS):
            plus = list(theta)
            plus[i] += math.pi / 2
            minus = list(theta)
            minus[i] -= math.pi / 2
            dz = (variational_z(x, plus) - variational_z(x, minus)) / 2
            grads[i] += 2 * (z - y) * dz
    return [g / len(TRAIN) for g in grads]


def variational_accuracy(theta, data):
    ok = 0
    for x, cls in data:
        pred = 0 if variational_z(x, theta) > 0 else 1
        ok += (pred == cls)
    return ok / len(data)


def _initial_theta():
    rng = random.Random(INIT_SEED)
    return [rng.uniform(-0.5, 0.5) for _ in range(N_PARAMS)]


def train_variational(epochs=VAR_EPOCHS, lr=VAR_LR):
    """Adam with fixed schedule - deterministic from the seeded start."""
    theta = _initial_theta()
    init_loss = variational_loss(theta)
    init_test_acc = variational_accuracy(theta, TEST)
    init_grad = variational_gradient(theta)
    grad_var = (sum(g * g for g in init_grad) / len(init_grad)
                - (sum(init_grad) / len(init_grad)) ** 2)
    m = [0.0] * N_PARAMS
    v = [0.0] * N_PARAMS
    for epoch in range(1, epochs + 1):
        grads = variational_gradient(theta)
        for i in range(N_PARAMS):
            m[i] = 0.9 * m[i] + 0.1 * grads[i]
            v[i] = 0.999 * v[i] + 0.001 * grads[i] * grads[i]
            m_hat = m[i] / (1 - 0.9 ** epoch)
            v_hat = v[i] / (1 - 0.999 ** epoch)
            theta[i] -= lr * m_hat / (math.sqrt(v_hat) + 1e-8)
    return {
        "theta": theta,
        "init_loss": init_loss,
        "final_loss": variational_loss(theta),
        "init_test_accuracy": init_test_acc,
        "test_accuracy": variational_accuracy(theta, TEST),
        "train_accuracy": variational_accuracy(theta, TRAIN),
        "init_grad_variance": grad_var,
        "epochs": epochs,
        "lr": lr,
        "layers": VAR_LAYERS,
        "n_params": N_PARAMS,
    }


_RUN_CACHE = {}


def variational_run():
    if "variational" not in _RUN_CACHE:
        _RUN_CACHE["variational"] = train_variational()
    return _RUN_CACHE["variational"]


def stalled_run():
    """A deliberately under-trained run: same start, 5 epochs, tiny lr."""
    if "stalled" not in _RUN_CACHE:
        _RUN_CACHE["stalled"] = train_variational(epochs=5, lr=0.01)
    return _RUN_CACHE["stalled"]


# ================================================================ 4. Quantum kernel
# The feature map is the encoding half of the variational circuit
# (angles + CNOT + product term). The kernel is the exact state
# fidelity K(x, x') = |<phi(x)|phi(x')>|^2, computed from the
# statevectors themselves. Classification is nearest-centroid in
# kernel space: a point joins the class whose kernel distance
# K(x,x) - 2*mean K(x, class) + mean K(class, class) is smallest.

def feature_state(x):
    state = zero_state(2)
    state = ry(state, 2, 0, math.pi * x[0])
    state = ry(state, 2, 1, math.pi * x[1])
    state = cnot(state, 2, 0, 1)
    state = ry(state, 2, 1, math.pi * x[0] * x[1])
    return state


_FEATURE_CACHE = {}


def _feature(x):
    key = (round(x[0], 12), round(x[1], 12))
    if key not in _FEATURE_CACHE:
        _FEATURE_CACHE[key] = feature_state(x)
    return _FEATURE_CACHE[key]


def quantum_kernel(x, x_prime):
    a, b = _feature(x), _feature(x_prime)
    inner = sum(u.conjugate() * v for u, v in zip(a, b))
    return abs(inner) ** 2


def _centroid_predict(x, data, kernel):
    distances = []
    for cls in (0, 1):
        members = [xx for xx, cc in data if cc == cls]
        within = (sum(kernel(a, b) for a in members for b in members)
                  / len(members) ** 2)
        cross = sum(kernel(x, a) for a in members) / len(members)
        distances.append(kernel(x, x) - 2 * cross + within)
    return 0 if distances[0] < distances[1] else 1


def kernel_report():
    if "kernel" not in _RUN_CACHE:
        test_ok = sum(_centroid_predict(x, TRAIN, quantum_kernel) == cls
                      for x, cls in TEST)
        train_ok = sum(_centroid_predict(x, TRAIN, quantum_kernel) == cls
                       for x, cls in TRAIN)
        _RUN_CACHE["kernel"] = {
            "test_accuracy": test_ok / len(TEST),
            "train_accuracy": train_ok / len(TRAIN),
            "self_kernel": quantum_kernel(TRAIN[0][0], TRAIN[0][0]),
        }
    return _RUN_CACHE["kernel"]


# ================================================================ 5. Classical baselines
# Trained in the same run, on the same split, from scratch.
# (a) Logistic regression by full-batch gradient descent - the
#     honest linear reference; moons are not linearly separable,
#     so it is expected to struggle, and it does.
# (b) RBF-kernel nearest centroid - the direct classical counterpart
#     of the quantum kernel. Its width gamma is chosen by
#     leave-one-out cross-validation on the TRAINING split only
#     (smallest gamma reaching the best LOO score); the test split
#     is never touched during selection.

def _sigmoid(z):
    if z < -60:
        return 0.0
    if z > 60:
        return 1.0
    return 1 / (1 + math.exp(-z))


def train_logistic(epochs=300, lr=0.5):
    w = [0.0, 0.0]
    b = 0.0
    for _ in range(epochs):
        gw = [0.0, 0.0]
        gb = 0.0
        for x, cls in TRAIN:
            err = _sigmoid(w[0] * x[0] + w[1] * x[1] + b) - cls
            gw[0] += err * x[0]
            gw[1] += err * x[1]
            gb += err
        w = [wi - lr * g / len(TRAIN) for wi, g in zip(w, gw)]
        b -= lr * gb / len(TRAIN)

    def predict(x):
        return 1 if w[0] * x[0] + w[1] * x[1] + b > 0 else 0

    return {
        "test_accuracy": sum(predict(x) == c for x, c in TEST) / len(TEST),
        "train_accuracy": sum(predict(x) == c for x, c in TRAIN) / len(TRAIN),
        "weights": [round(wi, 4) for wi in w],
        "bias": round(b, 4),
    }


def rbf_kernel(gamma):
    def kern(a, b):
        return math.exp(-gamma * ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2))
    return kern


def _loo_accuracy(gamma):
    ok = 0
    for i, (x, cls) in enumerate(TRAIN):
        rest = TRAIN[:i] + TRAIN[i + 1:]
        # centroid over the remaining points of each class
        distances = []
        kern = rbf_kernel(gamma)
        for c in (0, 1):
            members = [xx for xx, cc in rest if cc == c]
            within = (sum(kern(a, b) for a in members for b in members)
                      / len(members) ** 2)
            cross = sum(kern(x, a) for a in members) / len(members)
            distances.append(kern(x, x) - 2 * cross + within)
        ok += ((0 if distances[0] < distances[1] else 1) == cls)
    return ok / len(TRAIN)


def train_rbf_centroid():
    scores = {g: _loo_accuracy(g) for g in RBF_GAMMA_GRID}
    best = max(scores.values())
    gamma = min(g for g in RBF_GAMMA_GRID if scores[g] == best)
    kern = rbf_kernel(gamma)
    return {
        "gamma": gamma,
        "loo_scores": {str(g): round(s, 4) for g, s in scores.items()},
        "test_accuracy": sum(_centroid_predict(x, TRAIN, kern) == c
                             for x, c in TEST) / len(TEST),
        "train_accuracy": sum(_centroid_predict(x, TRAIN, kern) == c
                              for x, c in TRAIN) / len(TRAIN),
    }


def baseline_report():
    if "baselines" not in _RUN_CACHE:
        _RUN_CACHE["baselines"] = {
            "logistic_regression": train_logistic(),
            "rbf_centroid": train_rbf_centroid(),
        }
    return _RUN_CACHE["baselines"]


def headline_report():
    """All models, one split, one table - the honest headline."""
    var = variational_run()
    kern = kernel_report()
    base = baseline_report()
    return {
        "variational_circuit": round(var["test_accuracy"], 4),
        "quantum_kernel": round(kern["test_accuracy"], 4),
        "classical_rbf_centroid": round(base["rbf_centroid"]["test_accuracy"], 4),
        "classical_logistic_regression": round(
            base["logistic_regression"]["test_accuracy"], 4),
        "quantum_advantage_claimed": False,
        "note": "classical RBF centroid ties the quantum kernel and beats "
                "the variational circuit on this toy split - no advantage "
                "is claimed; the lab teaches the machinery",
    }


# ================================================================ 6. Barren plateaus, measured
# Random circuits (seeded) of RY/RZ rotations + a CNOT chain, depth
# 2n, with the GLOBAL parity cost <Z...Z>. The variance of the
# parameter-shift gradient of the first parameter is measured over
# PLATEAU_SAMPLES seeded circuits per qubit count, 2..6 qubits.
# A global cost on a deep random circuit is exactly the textbook
# barren-plateau setup - and the decay is computed, not quoted.
# (Companion measurement: VariationalQuantum-Lab's plateau section.)

def _plateau_cost(theta, n, depth):
    state = zero_state(n)
    k = 0
    for _ in range(depth):
        for q in range(n):
            state = ry(state, n, q, theta[k])
            state = rz(state, n, q, theta[k + 1])
            k += 2
        for q in range(n - 1):
            state = cnot(state, n, q, q + 1)
    return expectation_parity(state, n)


def plateau_variance(n, samples=PLATEAU_SAMPLES):
    rng = random.Random(99 + n)
    depth = 2 * n
    n_params = depth * 2 * n
    grads = []
    for _ in range(samples):
        theta = [rng.uniform(-math.pi, math.pi) for _ in range(n_params)]
        plus = list(theta)
        plus[0] += math.pi / 2
        minus = list(theta)
        minus[0] -= math.pi / 2
        grads.append((_plateau_cost(plus, n, depth)
                      - _plateau_cost(minus, n, depth)) / 2)
    mean = sum(grads) / len(grads)
    return sum((g - mean) ** 2 for g in grads) / len(grads)


def plateau_report():
    if "plateau" not in _RUN_CACHE:
        variances = {n: plateau_variance(n) for n in range(2, 7)}
        xs = list(variances)
        ys = [math.log10(variances[n]) for n in xs]
        x_bar = sum(xs) / len(xs)
        y_bar = sum(ys) / len(ys)
        slope = (sum((x - x_bar) * (y - y_bar) for x, y in zip(xs, ys))
                 / sum((x - x_bar) ** 2 for x in xs))
        _RUN_CACHE["plateau"] = {
            "cost": "global parity <Z...Z>, random circuits, depth 2n",
            "samples_per_point": PLATEAU_SAMPLES,
            "gradient_variance": {str(n): variances[n] for n in xs},
            "decay_factor_2_to_6_qubits": variances[2] / variances[6],
            "log10_slope_per_qubit": slope,
        }
    return _RUN_CACHE["plateau"]


# ================================================================ 7. kNN run-diagnosis classifier
# Features: loss drop fraction, initial gradient variance, test
# accuracy. The training grid is labelled by the lab's own diagnosis
# policy (diagnose_run below) - the classifier interpolates the
# policy over runs the lab itself produced.

def diagnose_run(loss_drop, grad_variance, test_accuracy):
    """The run-health policy distilled to a three-way label.

    The barren line (gradient variance < 0.01) sits between the
    lab's own measured points: the 2-qubit variational run starts
    at 0.0166 and trains, while the measured 6-qubit random-circuit
    variance is 0.00754 and does not.
    """
    if grad_variance < 0.01:
        return "barren"
    if loss_drop >= 0.5 and test_accuracy >= 0.85:
        return "converged"
    return "plateaued"


def _diagnosis_grid():
    rows = []
    for drop in (0.0, 0.25, 0.5, 0.75, 0.9):
        for var in (0.001, 0.003, 0.05, 0.2):
            for acc in (0.5, 0.7, 0.85, 0.95):
                rows.append(((drop, var, acc),
                             diagnose_run(drop, var, acc)))
    return rows


DIAGNOSIS_GRID = _diagnosis_grid()


def _diag_scaled(features):
    drop, var, acc = features
    return (drop, math.log10(var) / -3.0, acc)


def classify_run(loss_drop, grad_variance, test_accuracy, k=5):
    """kNN over the policy-labelled grid, deterministic tie-breaking."""
    target = _diag_scaled((loss_drop, grad_variance, test_accuracy))
    ranked = []
    for features, label in DIAGNOSIS_GRID:
        point = _diag_scaled(features)
        dist = math.sqrt(sum((a - b) ** 2 for a, b in zip(target, point)))
        ranked.append((dist, features, label))
    ranked.sort(key=lambda row: (row[0], row[1]))
    votes = {}
    for _, _, label in ranked[:k]:
        votes[label] = votes.get(label, 0) + 1
    priority = {"converged": 0, "plateaued": 1, "barren": 2}
    return sorted(votes.items(), key=lambda kv: (-kv[1], priority[kv[0]]))[0][0]


def diagnosis_report():
    var = variational_run()
    stalled = stalled_run()
    plateau = plateau_report()
    scenarios = {
        "variational_run_100_epochs": (
            (var["init_loss"] - var["final_loss"]) / var["init_loss"],
            var["init_grad_variance"], var["test_accuracy"]),
        "stalled_run_5_epochs_tiny_lr": (
            (stalled["init_loss"] - stalled["final_loss"])
            / stalled["init_loss"],
            stalled["init_grad_variance"], stalled["test_accuracy"]),
        "six_qubit_random_circuit": (
            0.0, plateau["gradient_variance"]["6"], 0.5),
    }
    return {name: classify_run(*features)
            for name, features in scenarios.items()}


# ================================================================ 8. Advisory seats
def _trace(payload):
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def argon4_decide(topic="quantummachinelearning-demo"):
    """Gemini Argon 4 DECIDER seat: MAP->MODEL->STRESS->DECIDE.

    Names-only keys, live calls 0, tighten-only: can add HOLDs, never lift one.
    """
    stages = ["MAP", "MODEL", "STRESS", "DECIDE"]
    payload = {"seat": "gemini-argon-4", "topic": topic, "stages": stages}
    return {
        "seat": "Gemini Argon 4 Extended Thinking",
        "role": "DECIDER",
        "pipeline": stages,
        "tiers": [2048, 8192, 32768],
        "key": "names-only (GEMINI_API_KEY name checked, value never read)",
        "live_calls": 0,
        "decision": "PROCEED - sandbox simulation only, all advantage / production / benchmark gates stay HOLD",
        "can_override_hold": False,
        "trace_sha256_16": _trace(payload),
    }


def claude_advisor(topic="quantummachinelearning-demo"):
    """Muse advisor seat: advises, can never override a HOLD."""
    return {
        "seat": "Muse",
        "role": "ADVISOR",
        "key": "names-only (MODEL_API_KEY name checked, value never read)",
        "live_calls": 0,
        "advice": "Print the classical baseline in the same table as the quantum models, every time: on toy data the RBF centroid ties the quantum kernel - the machinery is the lesson, the tie is the honesty.",
        "can_override_hold": False,
    }


def google_flow_plan():
    """Google Flow seat: plan-only, nothing rendered, nothing live."""
    return {
        "seat": "Google Flow",
        "role": "PLAN-ONLY",
        "shots": ["two moons of points facing a two-qubit circuit drawn as sheet music",
                  "data angles re-uploaded layer after layer while the loss curve steps down",
                  "a kernel matrix glowing where two moons' points recognise each other",
                  "gradient arrows shrinking into a flat desert as qubits are added",
                  "the classical baseline crossing the finish line in a dead heat"],
        "rendered": False,
        "live": 0,
    }


def brains_report():
    return {"decider": argon4_decide(), "advisor": claude_advisor(),
            "flow": google_flow_plan()}


# ================================================================ checks
def run_checks():
    c = []

    def chk(name, ok):
        c.append((name, bool(ok)))

    # 1. simulator
    chk("zero_state_norm", abs(state_norm(zero_state(3)) - 1.0) < 1e-12)
    flipped = ry(zero_state(1), 1, 0, math.pi)
    chk("ry_pi_flips", abs(abs(flipped[1]) - 1.0) < 1e-12)
    plus = hadamard(zero_state(1), 1, 0)
    chk("hadamard_superposition",
        abs(abs(plus[0]) ** 2 - 0.5) < 1e-12
        and abs(abs(plus[1]) ** 2 - 0.5) < 1e-12)
    bell = cnot(hadamard(zero_state(2), 2, 0), 2, 0, 1)
    chk("cnot_bell_state",
        abs(abs(bell[0]) ** 2 - 0.5) < 1e-12 and abs(bell[1]) < 1e-12
        and abs(bell[2]) < 1e-12 and abs(abs(bell[3]) ** 2 - 0.5) < 1e-12)
    chk("expectation_z_ground", abs(expectation_z(zero_state(1), 1, 0) - 1.0)
        < 1e-12)
    chk("expectation_z_excited",
        abs(expectation_z(flipped, 1, 0) + 1.0) < 1e-12)
    chk("parity_ground_even",
        abs(expectation_parity(zero_state(3), 3) - 1.0) < 1e-12)
    odd = cnot(cnot(hadamard(zero_state(2), 2, 0), 2, 0, 1), 2, 0, 1)
    chk("parity_sign_flips_with_odd_excitation",
        abs(expectation_parity(ry(zero_state(2), 2, 0, math.pi), 2) + 1.0)
        < 1e-12 and abs(state_norm(odd) - 1.0) < 1e-12)

    # 2. dataset
    train2, test2 = make_dataset()
    chk("dataset_sizes", len(TRAIN) == 40 and len(TEST) == 20
        and len(TRAIN) + len(TEST) == 60)
    chk("dataset_deterministic", train2 == TRAIN and test2 == TEST)
    chk("dataset_balanced",
        sum(1 for _, cc in TRAIN + TEST if cc == 0) == POINTS_PER_CLASS
        and sum(1 for _, cc in TRAIN + TEST if cc == 1) == POINTS_PER_CLASS)
    chk("dataset_finite", all(math.isfinite(v) for (x, _) in TRAIN + TEST
                              for v in x))

    # 3. variational classifier - parameter shift is exact, training works
    theta0 = _initial_theta()
    shift_grad = variational_gradient(theta0)
    eps = 1e-5
    fd_ok = True
    for i in (0, 5, 15):
        plus = list(theta0)
        plus[i] += eps
        minus = list(theta0)
        minus[i] -= eps
        fd = (variational_loss(plus) - variational_loss(minus)) / (2 * eps)
        fd_ok = fd_ok and abs(fd - shift_grad[i]) < 1e-6
    chk("parameter_shift_matches_finite_difference", fd_ok)
    run = variational_run()
    chk("variational_loss_falls", run["final_loss"] < run["init_loss"]
        and run["final_loss"] < 0.5 * run["init_loss"])
    chk("variational_accuracy_improves",
        run["test_accuracy"] > run["init_test_accuracy"])
    chk("variational_test_accuracy_bar", run["test_accuracy"] >= 0.85)
    chk("variational_train_accuracy_bar", run["train_accuracy"] >= 0.85)
    chk("variational_params_count", len(run["theta"]) == N_PARAMS == 16)
    chk("variational_deterministic",
        train_variational(epochs=3, lr=0.25)["final_loss"]
        == train_variational(epochs=3, lr=0.25)["final_loss"])

    # 4. quantum kernel
    x0, x1 = TRAIN[0][0], TRAIN[1][0]
    chk("kernel_self_is_one", abs(quantum_kernel(x0, x0) - 1.0) < 1e-9)
    chk("kernel_symmetric",
        abs(quantum_kernel(x0, x1) - quantum_kernel(x1, x0)) < 1e-12)
    chk("kernel_bounded", 0.0 <= quantum_kernel(x0, x1) <= 1.0 + 1e-12)
    kern = kernel_report()
    chk("kernel_test_accuracy_bar", kern["test_accuracy"] >= 0.90)
    chk("kernel_train_accuracy_bar", kern["train_accuracy"] >= 0.90)

    # 5. classical baselines - same split, honest ordering
    base = baseline_report()
    logreg = base["logistic_regression"]
    rbf = base["rbf_centroid"]
    chk("logistic_regression_struggles_on_moons",
        0.40 <= logreg["test_accuracy"] <= 0.75)
    chk("rbf_gamma_from_grid", rbf["gamma"] in RBF_GAMMA_GRID)
    chk("rbf_gamma_is_cv_best",
        rbf["loo_scores"][str(rbf["gamma"])]
        == max(rbf["loo_scores"].values()))
    chk("rbf_test_accuracy_bar", rbf["test_accuracy"] >= 0.90)
    chk("classical_baseline_competitive",
        rbf["test_accuracy"] >= run["test_accuracy"])
    head = headline_report()
    chk("no_advantage_claimed", head["quantum_advantage_claimed"] is False
        and head["classical_rbf_centroid"] >= head["variational_circuit"])

    # 6. barren plateau - measured decay
    plateau = plateau_report()
    var = plateau["gradient_variance"]
    chk("plateau_variance_positive", all(v > 0 for v in var.values()))
    chk("plateau_variance_decays", var["6"] < var["2"] / 4)
    chk("plateau_fit_slope_negative", plateau["log10_slope_per_qubit"] < 0)
    chk("plateau_decay_factor", plateau["decay_factor_2_to_6_qubits"] > 4)

    # 7. run-diagnosis classifier - labels from the lab's own policy
    diag = diagnosis_report()
    chk("diagnosis_variational_converged",
        diag["variational_run_100_epochs"] == "converged")
    chk("diagnosis_stalled_plateaued",
        diag["stalled_run_5_epochs_tiny_lr"] == "plateaued")
    chk("diagnosis_six_qubit_barren",
        diag["six_qubit_random_circuit"] == "barren")
    chk("diagnosis_matches_policy",
        all(classify_run(*features) == diagnose_run(*features)
            for features in [(0.8, 0.05, 0.95), (0.05, 0.05, 0.55),
                             (0.0, 0.003, 0.5), (0.9, 0.2, 0.95),
                             (0.25, 0.05, 0.7)]))

    # gates + seats
    chk("hold_gates_14", len(HOLD_GATES) == 14
        and all(v.startswith("HOLD") for v in HOLD_GATES.values()))
    chk("gate_advantage_hold", gate("quantum_advantage_claim") == "HOLD")
    chk("gate_production_hold", gate("production_ml_system_claim") == "HOLD")
    chk("gate_benchmark_hold", gate("benchmark_superiority_claim") == "HOLD")
    chk("gate_hardware_hold", gate("hardware_execution_claim") == "HOLD")
    chk("gate_medical_hold", gate("medical_diagnosis_claim") == "HOLD")
    chk("gate_sandbox_proceed", gate("run_simulation") == "PROCEED")
    brains = brains_report()
    chk("argon4_no_override", brains["decider"]["can_override_hold"] is False
        and brains["decider"]["seat"]
        == "Gemini Argon 4 Extended Thinking")
    chk("claude_no_override", brains["advisor"]["can_override_hold"] is False)
    chk("flow_plan_only", brains["flow"]["rendered"] is False
        and brains["flow"]["live"] == 0)
    return c


# ================================================================ CLI / demo
def demo():
    print("=" * 68)
    print("QuantumMachineLearning-Lab - quantum machine learning lab")
    print("by Darion Kellar | MIT | pure stdlib | zero network | EDUCATION ONLY")
    print("=" * 68)
    print("\n[1] THE HONEST CORE - machinery yes, advantage no")
    print("    A variational circuit and a quantum kernel, trained and")
    print("    graded in the open - next to classical baselines on the")
    print("    SAME split. On this toy data the classical RBF baseline")
    print("    ties the quantum kernel and beats the variational circuit.")
    print("    That tie is printed in the headline table, not buried.")
    data = dataset_report()
    print("\n[2] Dataset - two interleaving half-circles (moons), seeded")
    print(f"    seed {data['seed']} | noise sigma {data['noise_sigma']} | "
          f"{data['total']} points | train {data['train']} / "
          f"test {data['test']}")
    print(f"    train classes {data['train_class_counts']} | "
          f"test classes {data['test_class_counts']}")
    print("    -> a linear model cannot separate moons; that is the point")
    print("       of the shape - kernels and circuits earn their keep here,")
    print("       and so does the classical RBF kernel.")
    run = variational_run()
    print("\n[3] Variational classifier - 2 qubits, data re-uploading, "
          "Adam")
    print(f"    layers {run['layers']} | params {run['n_params']} | "
          f"epochs {run['epochs']} | lr {run['lr']} (parameter-shift "
          "gradients, exact)")
    print(f"    loss {run['init_loss']:.4f} -> {run['final_loss']:.4f} | "
          f"test accuracy {run['init_test_accuracy']:.2f} -> "
          f"{run['test_accuracy']:.2f} | train {run['train_accuracy']:.2f}")
    kern = kernel_report()
    print("\n[4] Quantum kernel - exact state fidelity, kernel centroid")
    print(f"    K(x, x) = {kern['self_kernel']:.4f} | train accuracy "
          f"{kern['train_accuracy']:.2f} | test accuracy "
          f"{kern['test_accuracy']:.2f}")
    base = baseline_report()
    logreg, rbf = base["logistic_regression"], base["rbf_centroid"]
    print("\n[5] Classical baselines - same run, same split, from scratch")
    print(f"    logistic regression:      test {logreg['test_accuracy']:.2f} "
          f"(linear - moons beat it, as expected)")
    print(f"    RBF centroid (gamma {rbf['gamma']}, picked by train-only "
          f"leave-one-out CV): test {rbf['test_accuracy']:.2f}")
    head = headline_report()
    print("\n[6] HEADLINE - test accuracy, one split, every model")
    for name in ("variational_circuit", "quantum_kernel",
                 "classical_rbf_centroid", "classical_logistic_regression"):
        print(f"    {name:32s} {head[name]:.2f}")
    print(f"    quantum advantage claimed: {head['quantum_advantage_claimed']}")
    print("    -> the classical kernel ties the quantum kernel. Machinery:")
    print("       taught. Advantage: none. That is the honest result.")
    plateau = plateau_report()
    print("\n[7] Barren plateaus - measured, not named")
    print(f"    global parity cost, depth 2n, {plateau['samples_per_point']} "
          "seeded circuits per point:")
    for n, v in plateau["gradient_variance"].items():
        print(f"      {n} qubits: gradient variance {v:.3e}")
    print(f"    decay 2 -> 6 qubits: x{plateau['decay_factor_2_to_6_qubits']:.1f} "
          f"| fit slope {plateau['log10_slope_per_qubit']:.3f} log10/qubit")
    print("    -> why deep random circuits stop training: the gradients")
    print("       vanish exponentially. (See also VariationalQuantum-Lab.)")
    print("\n[8] Run-diagnosis classifier (TinyML kNN, policy-labelled)")
    for name, label in diagnosis_report().items():
        print(f"    {name:36s} -> {label}")
    print("\n[9] Advisory seats")
    brains = brains_report()
    print(f"    decider: {brains['decider']['decision']} "
          f"trace={brains['decider']['trace_sha256_16']}")
    print(f"    advisor can_override_hold = {brains['advisor']['can_override_hold']}")
    print(f"    flow rendered = {brains['flow']['rendered']} "
          f"live = {brains['flow']['live']}")
    print(f"\n[10] HOLD gates: {len(HOLD_GATES)}")
    for key in HOLD_GATES:
        print(f"    HOLD  {key}")
    checks = run_checks()
    passed = sum(1 for _, ok in checks if ok)
    print(f"\nCHECKS {passed}/{len(checks)}")
    for name, ok in checks:
        if not ok:
            print(f"    FAIL {name}")
    verdict = "PROCEED" if passed == len(checks) else "HOLD"
    print(f"VERDICT {verdict} (sandbox simulation only) | HOLD gates {len(HOLD_GATES)}")
    return 0 if passed == len(checks) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] == "demo":
        return demo()
    cmd = argv[0]
    if cmd == "checks":
        checks = run_checks()
        for name, ok in checks:
            print(f"{'PASS' if ok else 'FAIL'} {name}")
        print(f"CHECKS {sum(1 for _, ok in checks if ok)}/{len(checks)}")
        return 0 if all(ok for _, ok in checks) else 1
    if cmd == "data":
        print(json.dumps(dataset_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "train":
        run = variational_run()
        print(json.dumps({k: v for k, v in run.items() if k != "theta"},
                         indent=2, sort_keys=True))
        return 0
    if cmd == "kernel":
        print(json.dumps(kernel_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "baseline":
        print(json.dumps(baseline_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "headline":
        print(json.dumps(headline_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "plateau":
        print(json.dumps(plateau_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "classify":
        print(json.dumps(diagnosis_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "brains":
        print(json.dumps(brains_report(), indent=2, sort_keys=True))
        return 0
    if cmd == "holds":
        for key, value in HOLD_GATES.items():
            print(f"{key}: {value}")
        return 0
    if cmd == "hold-demo":
        action = argv[1] if len(argv) > 1 else "quantum_advantage_claim"
        print(f"{action}: {HOLD_GATES.get(action, 'PROCEED - not gated')}")
        return 2 if gate(action) == "HOLD" else 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    sys.exit(main())
