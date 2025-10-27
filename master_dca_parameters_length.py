"""Python port of ``masterDCAparameters_length.m`` from the DCA-scape repo."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from fastahamiltonian import Alignment, letter2number


def master_dca_parameters_length(inputfile: str | Path, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Estimate DCA couplings and local fields averaged over four gauges."""

    couplings: List[np.ndarray] = []
    local_fields: List[np.ndarray] = []
    for rtype in range(1, 5):
        h, c, _ = dca_parameters_1(inputfile, stype=2, rtype=rtype)
        local_fields.append(h)
        couplings.append(c)

    c_avg, h_avg = average_couplings_localfields(
        couplings[0],
        couplings[1],
        couplings[2],
        couplings[3],
        local_fields[0],
        local_fields[1],
        local_fields[2],
        local_fields[3],
        length,
    )
    return c_avg, h_avg


def average_couplings_localfields(
    family1: np.ndarray,
    family2: np.ndarray,
    family3: np.ndarray,
    family4: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
    h3: np.ndarray,
    h4: np.ndarray,
    length: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Average couplings and local fields across the four nucleotide gauges."""

    q = 4
    dim = q * length

    c_4_1 = np.zeros((dim, dim))
    c_4_2 = np.zeros((dim, dim))
    c_4_3 = np.zeros((dim, dim))

    for i in range(0, dim, q):
        for j in range(0, dim, q):
            c_4_1[i, j] = family1[i + 3, j + 3]
            c_4_1[i, j + 1] = family1[i + 3, j + 1]
            c_4_1[i, j + 2] = family1[i + 3, j + 2]
            c_4_1[i, j + 3] = family1[i + 3, j]

            c_4_1[i + 1, j] = family1[i + 1, j + 3]
            c_4_1[i + 1, j + 1] = family1[i + 1, j + 1]
            c_4_1[i + 1, j + 2] = family1[i + 1, j + 2]
            c_4_1[i + 1, j + 3] = family1[i + 1, j]

            c_4_1[i + 2, j] = family1[i + 2, j + 3]
            c_4_1[i + 2, j + 1] = family1[i + 2, j + 1]
            c_4_1[i + 2, j + 2] = family1[i + 2, j + 2]
            c_4_1[i + 2, j + 3] = family1[i + 2, j]

            c_4_1[i + 3, j] = family1[i, j + 3]
            c_4_1[i + 3, j + 1] = family1[i, j + 1]
            c_4_1[i + 3, j + 2] = family1[i, j + 2]
            c_4_1[i + 3, j + 3] = family1[i, j]

            c_4_2[i, j] = family2[i, j]
            c_4_2[i, j + 1] = family2[i, j + 3]
            c_4_2[i, j + 2] = family2[i, j + 2]
            c_4_2[i, j + 3] = family2[i, j + 1]

            c_4_2[i + 1, j] = family2[i + 3, j]
            c_4_2[i + 1, j + 1] = family2[i + 3, j + 3]
            c_4_2[i + 1, j + 2] = family2[i + 3, j + 2]
            c_4_2[i + 1, j + 3] = family2[i + 3, j + 1]

            c_4_2[i + 2, j] = family2[i + 2, j]
            c_4_2[i + 2, j + 1] = family2[i + 2, j + 3]
            c_4_2[i + 2, j + 2] = family2[i + 2, j + 2]
            c_4_2[i + 2, j + 3] = family2[i + 2, j + 1]

            c_4_2[i + 3, j] = family2[i + 1, j]
            c_4_2[i + 3, j + 1] = family2[i + 1, j + 3]
            c_4_2[i + 3, j + 2] = family2[i + 1, j + 2]
            c_4_2[i + 3, j + 3] = family2[i + 1, j + 1]

            c_4_3[i, j] = family3[i, j]
            c_4_3[i, j + 1] = family3[i, j + 1]
            c_4_3[i, j + 2] = family3[i, j + 3]
            c_4_3[i, j + 3] = family3[i, j + 2]

            c_4_3[i + 1, j] = family3[i + 1, j]
            c_4_3[i + 1, j + 1] = family3[i + 1, j + 1]
            c_4_3[i + 1, j + 2] = family3[i + 1, j + 3]
            c_4_3[i + 1, j + 3] = family3[i + 1, j + 2]

            c_4_3[i + 2, j] = family3[i + 3, j]
            c_4_3[i + 2, j + 1] = family3[i + 3, j + 1]
            c_4_3[i + 2, j + 2] = family3[i + 3, j + 3]
            c_4_3[i + 2, j + 3] = family3[i + 3, j + 2]

            c_4_3[i + 3, j] = family3[i + 2, j]
            c_4_3[i + 3, j + 1] = family3[i + 2, j + 1]
            c_4_3[i + 3, j + 2] = family3[i + 2, j + 3]
            c_4_3[i + 3, j + 3] = family3[i + 2, j + 2]

    c_average = (c_4_1 + c_4_2 + c_4_3 + family4) / 4.0

    h_4_1 = np.zeros((4, length))
    h_4_2 = np.zeros((4, length))
    h_4_3 = np.zeros((4, length))

    for i in range(length):
        h_4_1[0, i] = h1[3, i]
        h_4_1[1, i] = h1[1, i]
        h_4_1[2, i] = h1[2, i]
        h_4_1[3, i] = h1[0, i]

        h_4_2[0, i] = h2[0, i]
        h_4_2[1, i] = h2[3, i]
        h_4_2[2, i] = h2[2, i]
        h_4_2[3, i] = h2[0, i]

        h_4_3[0, i] = h3[0, i]
        h_4_3[1, i] = h3[1, i]
        h_4_3[2, i] = h3[3, i]
        h_4_3[3, i] = h3[2, i]

    h_average = (h_4_1 + h_4_2 + h_4_3 + h4) / 4.0
    return c_average, h_average


def dca_parameters_1(
    inputfile: str | Path,
    stype: int,
    rtype: int,
    pseudocount_weight: float = 0.5,
    theta: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Python port of the MATLAB ``DCAparameters_1`` routine."""

    alignment = return_alignment(inputfile, stype, rtype)
    n = alignment.length
    m = alignment.size
    q = alignment.alphabet
    align = alignment.sequences

    if theta > 0.0:
        weights = compute_sequence_weights(align, theta)
    else:
        weights = np.ones(m)

    meff = weights.sum()
    pij_true, pi_true = compute_true_frequencies(align, weights, meff, n, q)
    pij, pi = with_pc(pij_true, pi_true, pseudocount_weight, n, q)
    c = compute_c(pij, pi, n, q)
    inv_c = np.linalg.inv(c)
    familycouplings = nicematrix(-inv_c, q)
    pairwise_h = compute_results(pi, inv_c, n, q)
    pairwise_h = symmetriclocal(pairwise_h, n, q)
    h, _ = averagehfield(pairwise_h)
    return h, familycouplings, align


@dataclass(frozen=True)
class AlignmentWithGauge(Alignment):
    gauge: int


def return_alignment(inputfile: str | Path, stype: int, rtype: int) -> AlignmentWithGauge:
    sequences = _read_fasta_sequences(inputfile)
    if not sequences:
        raise ValueError("input FASTA file does not contain any sequences")

    mask = [
        (char != "." and char == char.upper())
        for char in sequences[0]
    ]

    filtered: List[List[int]] = []
    for seq in sequences:
        if len(seq) != len(mask):
            raise ValueError("all sequences in the FASTA file must have equal length")
        converted: List[int] = []
        for keep, char in zip(mask, seq):
            if keep:
                converted.append(letter2number_rtype(char.upper(), stype, rtype))
        filtered.append(converted)

    array = np.asarray(filtered, dtype=int)
    q = int(array.max(initial=1))
    length = array.shape[1]
    size = array.shape[0]
    return AlignmentWithGauge(length=length, size=size, alphabet=q, sequences=array, gauge=rtype)


def _read_fasta_sequences(path: str | Path) -> List[str]:
    sequences: List[str] = []
    current: List[str] = []
    with open(Path(path), "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current).upper())
                    current.clear()
                continue
            current.append(line)
    if current:
        sequences.append("".join(current).upper())
    return sequences


def letter2number_rtype(char: str, stype: int, rtype: int) -> int:
    if stype == 1:
        return letter2number(char, stype)

    if stype != 2:
        raise ValueError("stype must be either 1 (protein) or 2 (RNA/DNA)")

    if rtype == 1:
        mapping = {"A": 4, "C": 2, "G": 3, "T": 1, "U": 1, "-": 5}
        return mapping.get(char, 4)
    if rtype == 2:
        mapping = {"A": 1, "C": 4, "G": 3, "T": 2, "U": 2, "-": 5}
        return mapping.get(char, 1)
    if rtype == 3:
        mapping = {"A": 1, "C": 2, "G": 4, "T": 3, "U": 3, "-": 5}
        return mapping.get(char, 1)
    if rtype == 4:
        mapping = {"A": 1, "C": 2, "G": 3, "T": 4, "U": 4, "-": 5}
        return mapping.get(char, 1)

    raise ValueError("rtype must be in {1, 2, 3, 4}")


def compute_sequence_weights(align: np.ndarray, theta: float) -> np.ndarray:
    m = align.shape[0]
    if theta <= 0.0:
        return np.ones(m)

    weights = np.ones(m)
    for seq in range(m):
        for other in range(m):
            if seq == other:
                continue
            distance = np.mean(align[seq] != align[other])
            if distance < theta:
                weights[seq] += 1.0
        weights[seq] = 1.0 / weights[seq]
    return weights


def compute_true_frequencies(
    align: np.ndarray,
    weights: np.ndarray,
    meff: float,
    n: int,
    q: int,
) -> tuple[np.ndarray, np.ndarray]:
    pij_true = np.zeros((n, n, q, q))
    pi_true = np.zeros((n, q))

    for seq_idx, weight in enumerate(weights):
        for i in range(n):
            symbol = align[seq_idx, i]
            pi_true[i, symbol - 1] += weight

    pi_true /= meff

    for seq_idx, weight in enumerate(weights):
        for i in range(n - 1):
            sym_i = align[seq_idx, i]
            for j in range(i + 1, n):
                sym_j = align[seq_idx, j]
                pij_true[i, j, sym_i - 1, sym_j - 1] += weight
                pij_true[j, i, sym_j - 1, sym_i - 1] = pij_true[i, j, sym_i - 1, sym_j - 1]

    pij_true /= meff

    for i in range(n):
        pij_true[i, i, :, :] = np.diag(pi_true[i])
    return pij_true, pi_true


def with_pc(
    pij_true: np.ndarray,
    pi_true: np.ndarray,
    pseudocount_weight: float,
    n: int,
    q: int,
) -> tuple[np.ndarray, np.ndarray]:
    pij = (1.0 - pseudocount_weight) * pij_true + pseudocount_weight / (q * q) * np.ones_like(pij_true)
    pi = (1.0 - pseudocount_weight) * pi_true + pseudocount_weight / q * np.ones_like(pi_true)

    eye = np.eye(q)
    for i in range(n):
        for alpha in range(q):
            for beta in range(q):
                pij[i, i, alpha, beta] = (
                    (1.0 - pseudocount_weight) * pij_true[i, i, alpha, beta]
                    + pseudocount_weight / q * eye[alpha, beta]
                )
    return pij, pi


def compute_c(pij: np.ndarray, pi: np.ndarray, n: int, q: int) -> np.ndarray:
    c = np.zeros((n * (q - 1), n * (q - 1)))
    for i in range(n):
        for j in range(n):
            for alpha in range(q - 1):
                for beta in range(q - 1):
                    c[mapkey(i, alpha, q), mapkey(j, beta, q)] = (
                        pij[i, j, alpha, beta] - pi[i, alpha] * pi[j, beta]
                    )
    return c


def mapkey(i: int, alpha: int, q: int) -> int:
    return (q - 1) * i + alpha


def compute_results(pi: np.ndarray, inv_c: np.ndarray, n: int, q: int) -> np.ndarray:
    pairwise = np.zeros((n * q, 2 * n))
    for i in range(n):
        for j in range(i + 1, n):
            w_mf = return_w(inv_c, i, j, q)
            start = 2 * j - 2
            pairwise[i * q : (i + 1) * q, start : start + 2] = bp_link(i, j, w_mf, pi, q)
    return pairwise


def return_w(inv_c: np.ndarray, i: int, j: int, q: int) -> np.ndarray:
    w = np.ones((q, q))
    rows = [mapkey(i, alpha, q) for alpha in range(q - 1)]
    cols = [mapkey(j, beta, q) for beta in range(q - 1)]
    w[: q - 1, : q - 1] = np.exp(-inv_c[np.ix_(rows, cols)])
    return w


def bp_link(i: int, j: int, w: np.ndarray, pi: np.ndarray, q: int) -> np.ndarray:
    mu1, mu2 = compute_mu(i, j, w, pi, q)
    mu1 = mu1 / mu1[-1]
    mu2 = mu2 / mu2[-1]
    mu1 = np.clip(mu1, 1e-12, None)
    mu2 = np.clip(mu2, 1e-12, None)
    return np.column_stack((np.log(mu1), np.log(mu2)))


def compute_mu(i: int, j: int, w: np.ndarray, pi: np.ndarray, q: int) -> tuple[np.ndarray, np.ndarray]:
    epsilon = 1e-4
    diff = 1.0
    mu1 = np.full(q, 1.0 / q)
    mu2 = np.full(q, 1.0 / q)
    pi_i = pi[i]
    pi_j = pi[j]

    while diff > epsilon:
        scra1 = mu2 @ w.T
        scra2 = mu1 @ w
        new1 = pi_i / scra1
        new1 /= new1.sum()
        new2 = pi_j / scra2
        new2 /= new2.sum()
        diff = max(np.max(np.abs(new1 - mu1)), np.max(np.abs(new2 - mu2)))
        mu1 = new1
        mu2 = new2
    return mu1, mu2


def symmetriclocal(localfield: np.ndarray, n: int, q: int) -> np.ndarray:
    symm = np.array(localfield, copy=True)
    for i in range(n):
        for j in range(i + 1, n):
            symm[j * q : (j + 1) * q, 2 * i : 2 * i + 2] = np.column_stack(
                (
                    localfield[i * q : (i + 1) * q, 2 * j - 1],
                    localfield[i * q : (i + 1) * q, 2 * j - 2],
                )
            )
    return symm


def averagehfield(pairwise: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = pairwise.shape[1] // 2
    q = pairwise.shape[0] // n
    hi = np.zeros((q, n))
    sigma = np.zeros((q, n))

    for i in range(n):
        if i == 0:
            subset = pairwise[0:q, 2: 2 * n: 2]
        elif i == n - 1:
            subset = pairwise[(n - 1) * q : n * q, 0 : 2 * (n - 1) : 2]
        else:
            left = pairwise[i * q : (i + 1) * q, 0 : 2 * i : 2]
            right = pairwise[i * q : (i + 1) * q, 2 * i : 2 * n : 2]
            subset = np.concatenate((left, right), axis=1)

        hi[:, i] = subset.mean(axis=1)
        sigma[:, i] = subset.std(axis=1, ddof=0)

    return hi, sigma


def nicematrix(familycouplings: np.ndarray, q: int) -> np.ndarray:
    n = familycouplings.shape[0] // (q - 1)
    coupling = np.zeros((q * n, q * n))
    for i in range(n):
        for j in range(n):
            ii = slice(i * (q - 1), (i + 1) * (q - 1))
            jj = slice(j * (q - 1), (j + 1) * (q - 1))
            newii = slice(i * q, i * q + q - 1)
            newjj = slice(j * q, j * q + q - 1)
            coupling[newii, newjj] = familycouplings[ii, jj]
    return coupling


__all__ = [
    "master_dca_parameters_length",
    "dca_parameters_1",
    "average_couplings_localfields",
]

