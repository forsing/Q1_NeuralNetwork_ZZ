#!/usr/bin/env python3
"""
Q1 Neural Network — (čisto kvantno, bez klasičnog treniranja):
  QNN = FIKSNO parametrizovano kolo (ZZFeatureMap + RealAmplitudes).
  Inputs (ulazi) i Weights (težine) se DETERMINISTIČKI izvode iz CELOG CSV-a.
  Forward pass: Statevector → Born verovatnoće → NEXT rastuća sedmorka ∈ {1..39}.
  Seed = 39; isti start uvek daje isti rezultat.

Okruženje: Python 3.11.13, qiskit 1.4.4, qiskit-machine-learning 0.8.3, macOS M1.
"""

from __future__ import annotations

import csv
import random
import warnings
from pathlib import Path
from typing import List, Tuple

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np

from qiskit import QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import Statevector

# =========================
# Seed za reproduktivnost
# =========================
SEED = 39
np.random.seed(SEED)
random.seed(SEED)
try:
    from qiskit_machine_learning.utils import algorithm_globals

    algorithm_globals.random_seed = SEED
except ImportError:
    pass

# =========================
# Konfiguracija
# =========================
CSV_PATH = Path("/Users/4c/Desktop/GHQ/data/loto7hh_4600_k31.csv")
N_QUBITS = 7        # 7 kolona iz sedmorke -> 7 input parametara; 2^7 = 128 stanja  (default)
N_NUMBERS = 7
N_MAX = 39
REPS_FM = 2         # dubina ZZFeatureMap (encoder)                                 (default)
REPS_ANS = 2        # dubina RealAmplitudes (ansatz — "sloj" QNN-a)                  (default)

# Deterministička grid-optimizacija hiperparametara (bez iterativne klasične petlje):
#   mera = kosinusna sličnost bias[0..38] (iz Born verovatnoća preko mod-39 kanti)
#          i normalizovanog histograma 1..39 iz CELOG CSV-a (što veće — to bolje).
GRID_NQ = (4, 5, 6, 7, 8)
GRID_REPS_FM = (1, 2, 3)
GRID_REPS_ANS = (1, 2, 3)
GRID_ENT = ("linear", "circular", "full")


# =========================
# CSV
# =========================
def load_rows(path: Path) -> np.ndarray:
    rows: List[List[int]] = []
    with open(path, newline="", encoding="utf-8") as f:
        r = csv.reader(f)
        header = next(r)
        if not header or "Num1" not in header[0]:
            f.seek(0)
            r = csv.reader(f)
            next(r, None)
        for row in r:
            if not row or row[0].strip() == "Num1":
                continue
            rows.append([int(row[i]) for i in range(N_NUMBERS)])
    return np.array(rows, dtype=int)


# =========================
# Deterministički inputs i weights iz CELOG CSV-a
# =========================
def freq_vector(H: np.ndarray) -> np.ndarray:
    """Histogram pojavljivanja brojeva 1..39 u celom H."""
    c = np.zeros(N_MAX, dtype=np.float64)
    for v in H.ravel():
        if 1 <= v <= N_MAX:
            c[int(v) - 1] += 1.0
    return c


def inputs_from_csv(H: np.ndarray, n_in: int) -> np.ndarray:
    """Ulaz QNN-a: srednja vrednost po 7 kolona celog CSV-a, skalirano u [0, π]."""
    col_mean = H.astype(np.float64).mean(axis=0)  # duž. 7
    v = col_mean[:n_in] if n_in <= col_mean.size else np.resize(col_mean, n_in)
    return (v - 1.0) / (N_MAX - 1.0) * np.pi


def weights_from_csv(H: np.ndarray, n_w: int) -> np.ndarray:
    """Težine QNN-a: normalizovani histogram 1..39, cikličan odabir, u [0, π]."""
    f = freq_vector(H)
    denom = max(float(f.max() - f.min()), 1e-12)
    fn = (f - f.min()) / denom
    w = np.array([fn[i % N_MAX] for i in range(n_w)], dtype=np.float64)
    return w * np.pi


# =========================
# QNN kolo: ZZFeatureMap + RealAmplitudes
# =========================
def build_qnn_circuit(nq: int, reps_fm: int = REPS_FM, reps_ans: int = REPS_ANS, ent: str = "linear"):
    try:
        from qiskit.circuit.library import zz_feature_map

        fm = zz_feature_map(nq, reps=reps_fm, entanglement=ent)
    except (ImportError, TypeError, AttributeError):
        from qiskit.circuit.library import ZZFeatureMap

        fm = ZZFeatureMap(feature_dimension=nq, reps=reps_fm, entanglement=ent)
    ans = RealAmplitudes(num_qubits=nq, reps=reps_ans, entanglement=ent)
    qc = QuantumCircuit(nq)
    qc.compose(fm, inplace=True)
    qc.compose(ans, inplace=True)
    return qc, list(fm.parameters), list(ans.parameters)


def forward_probs(qc: QuantumCircuit, in_params, w_params, x: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Forward pass QNN-a preko tačnog Statevector-a (deterministički, bez uzorkovanja)."""
    bind = {p: float(v) for p, v in zip(in_params, x)}
    bind.update({p: float(v) for p, v in zip(w_params, w)})
    bound = qc.assign_parameters(bind)
    sv = Statevector(bound)
    return np.abs(sv.data) ** 2


# =========================
# Readout: NEXT sedmorka iz 2^n stanja → 39 kanti (mod), top-7 rastuće
# =========================
def bias_39(probs: np.ndarray, n_max: int = N_MAX) -> np.ndarray:
    b = np.zeros(n_max, dtype=np.float64)
    for idx, p in enumerate(probs):
        b[idx % n_max] += float(p)
    s = float(b.sum())
    return b / s if s > 0 else b


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-18 or nb < 1e-18:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def pick_next_combination(probs: np.ndarray, k: int = N_NUMBERS, n_max: int = N_MAX) -> Tuple[int, ...]:
    b = bias_39(probs, n_max)
    order = np.argsort(-b, kind="stable")
    out = sorted(int(o + 1) for o in order[:k])
    return tuple(out)


# =========================
# Determ. grid-optimizacija (nq, reps_fm, reps_ans, ent) po meri cos(bias, freq)
# =========================
def optimize_hparams(H: np.ndarray):
    f_csv = freq_vector(H)
    f_csv_n = f_csv / float(f_csv.sum() or 1.0)
    best = None
    for nq in GRID_NQ:
        for rfm in GRID_REPS_FM:
            for rans in GRID_REPS_ANS:
                for ent in GRID_ENT:
                    try:
                        qc, inp, wp = build_qnn_circuit(nq, rfm, rans, ent)
                        x = inputs_from_csv(H, len(inp))
                        w = weights_from_csv(H, len(wp))
                        probs = forward_probs(qc, inp, wp, x, w)
                        s = float(probs.sum())
                        if s > 0:
                            probs = probs / s
                        b = bias_39(probs)
                        score = cosine(b, f_csv_n)
                    except Exception:
                        continue
                    key = (score, -nq, -rfm, -rans, ent)
                    if best is None or key > best[0]:
                        best = (key, dict(nq=nq, reps_fm=rfm, reps_ans=rans, ent=ent, score=score))
    return best[1] if best else None


def main() -> int:
    warnings.filterwarnings("ignore", category=DeprecationWarning, message=r".*ZZFeatureMap.*")

    H = load_rows(CSV_PATH)
    if H.shape[0] < 1:
        print("premalo redova")
        return 1

    print("Q1 NN (A): CSV:", CSV_PATH)
    print("redova:", H.shape[0], "| seed:", SEED)

    best = optimize_hparams(H)
    if best is None:
        print("grid optimizacija nije uspela")
        return 2
    print(
        "BEST hparam:",
        "nq=", best["nq"],
        "| reps FM/Ans:", best["reps_fm"], best["reps_ans"],
        "| ent:", best["ent"],
        "| cos(bias, freq_csv):", round(float(best["score"]), 6),
    )

    qc, in_params, w_params = build_qnn_circuit(
        best["nq"], best["reps_fm"], best["reps_ans"], best["ent"]
    )
    x = inputs_from_csv(H, len(in_params))
    w = weights_from_csv(H, len(w_params))

    print("inputs (π-skala):", tuple(round(float(a), 4) for a in x))
    print("n_weights:", len(w))

    probs = forward_probs(qc, in_params, w_params, x, w)
    s = float(probs.sum())
    if s > 0:
        probs = probs / s

    pred = pick_next_combination(probs)
    print("predikcija NEXT:", pred)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())



"""
Q1 NN: CSV: /data/loto7hh_4600_k31.csv
redova: 4600 | seed: 39
BEST hparam: nq= 7 | reps FM/Ans: 1 1 | ent: linear | cos(bias, freq_csv): 0.951157
inputs (π-skala): (0.3433, 0.7474, 1.1585, 1.5756, 1.9892, 2.4074, 2.8166)
n_weights: 14
predikcija NEXT: (2, 6, 11, 14, 18, 21, 23)
"""



"""
Q1_NeuralNetwork_ZZ.py — QNN kao fiksni forward pass 

Učita CEO CSV, izračuna deterministčke ulaze i težine iz statistika CSV-a.
Izgradi QNN kolo: ZZFeatureMap (encoder) + RealAmplitudes (ansatz), bez klasičnog treniranja.
Forward pass: tačan Statevector → Born verovatnoće → bias_39 (mod-39) → NEXT rastuća sedmorka iz {1..39}.
Deterministička grid-optimizacija hiperparametara (nq, reps_fm, reps_ans, entanglement) po meri cos(bias, freq_csv).

Angle-encoding preko ZZFeatureMap (drugog reda, entanglement između qubit-a).
Varijacioni ansatz RealAmplitudes — ali sa fiksiranim težinama, ne trenirano.
Egzaktni simulator (Statevector) — bez šuma, bez uzorkovanja.
Grid-search optimizacija, kosinusna sličnost sa istorijskom frekvencijom.

Prednosti:
Čisto kvantno, bez klasičnih optimizera.
Reproduktivno: isti izlaz uvek.
Brz (Statevector je egzaktan, bez shots).
Jednostavna, standardna QNN arhitektura iz Qiskit-a.
Svi parametri (nq, dubine, entanglement) se biraju grid-scan-om — nema ručnog tjunera.

Nedostaci:
Težine nisu naučene iz zavisnosti uzastopnih redova (ne modeluje sekvencu, samo histogram).
Mera optimizacije je cos sa frekvencijom — to u suštini vodi model ka tome da reprodukuje marginalu, a ne novi signal.
mod-39 mapiranje 2^n → 39 je nasilno svođenje, može brisati fino razlikovanje stanja.
ZZFeatureMap/RealAmplitudes su generička kola — nisu prilagođena strukturi loto izvlačenja.
Eksponencijalno po nq za Statevector; u gridu sve preko 10 qubit-a nije praktično.
Predikcija je dominirana statistikom „čestih brojeva“ iz celog CSV-a, što u praksi daje stabilan ali konzervativan izbor.
"""
