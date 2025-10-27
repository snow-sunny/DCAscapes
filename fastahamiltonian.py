"""Python implementation of the ``Fastahamiltonian`` MATLAB routine.

This module mirrors the behaviour of ``Fastahamiltonian.m`` that ships with
the original DCA-scape repository.  The public ``fastahamiltonian`` function
exposes the same inputs and produces the same Hamiltonian scores as the
MATLAB code, but operates on :mod:`numpy` arrays and plain Python types.

The implementation follows the structure of the MATLAB source closely so that
the numerical results remain drop-in compatible.  Only minimal, explicit
Python-specific conveniences (type hints, docstrings, error checking) were
added.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np


@dataclass(frozen=True)
class Alignment:
    """Simple container returned by :func:`return_alignment`.

    Attributes
    ----------
    length:
        Number of retained positions per sequence after removing inserts.
    size:
        Number of sequences contained in the alignment.
    alphabet:
        Cardinality ``q`` of the Potts alphabet induced by the alignment.
    sequences:
        Integer encoded alignment with shape ``(size, length)`` where every
        entry is in ``[1, q]`` as in the MATLAB implementation.
    """

    length: int
    size: int
    alphabet: int
    sequences: np.ndarray


def fastahamiltonian(
    inputfile: str | Path,
    couplings: np.ndarray,
    localfields: np.ndarray,
    htype: int,
    n1: int,
    stype: int,
) -> np.ndarray:
    """Compute Potts Hamiltonians for every sequence in a FASTA alignment.

    Parameters
    ----------
    inputfile:
        Path to the FASTA file that contains the alignment.
    couplings:
        Coupling matrix with shape ``(q * N, q * N)`` as produced by
        :func:`master_dca_parameters_length.master_dca_parameters_length`.
    localfields:
        Local fields matrix with shape ``(q, N)`` using the same ``q`` and
        ``N`` as the alignment.
    htype:
        Hamiltonian type (1 or 2).  Type 1 sums couplings across two species
        while type 2 evaluates the full Potts Hamiltonian.
    n1:
        Length of the first species; only used when ``htype`` equals 1.
    stype:
        Species type: 1 for proteins, 2 for RNA/DNA.

    Returns
    -------
    numpy.ndarray
        Array of Hamiltonian scores (length equals the number of sequences).
    """

    alignment = return_alignment(inputfile, stype)
    sequences = alignment.sequences
    num_sequences, num_positions = sequences.shape
    q = alignment.alphabet

    if localfields.shape != (q, num_positions):
        raise ValueError(
            "localfields shape is incompatible with alignment: "
            f"expected {(q, num_positions)} got {localfields.shape}"
        )
    if couplings.shape != (q * num_positions, q * num_positions):
        raise ValueError(
            "couplings shape is incompatible with alignment: "
            f"expected {(q * num_positions, q * num_positions)} got {couplings.shape}"
        )

    h = np.zeros(num_sequences, dtype=float)

    # Local fields term.
    column_indices = np.arange(num_positions)
    for seq_idx in range(num_sequences):
        residue_indices = sequences[seq_idx, :] - 1  # convert to 0-based
        h[seq_idx] += localfields[residue_indices, column_indices].sum()

    # Coupling term.
    if htype == 1:
        if not (0 < n1 <= num_positions):
            raise ValueError("n1 must satisfy 0 < n1 <= number of positions")
        for seq_idx in range(num_sequences):
            seq = sequences[seq_idx]
            for res in range(n1):
                iindex_offset = q * res
                iindex = iindex_offset + seq[res] - 1
                for pair in range(n1, num_positions):
                    jindex = q * pair + seq[pair] - 1
                    h[seq_idx] += couplings[iindex, jindex]
    elif htype == 2:
        for seq_idx in range(num_sequences):
            seq = sequences[seq_idx]
            for res in range(num_positions - 1):
                iindex_offset = q * res
                iindex = iindex_offset + seq[res] - 1
                for pair in range(res + 1, num_positions):
                    jindex = q * pair + seq[pair] - 1
                    h[seq_idx] += couplings[iindex, jindex]
    else:
        raise ValueError("htype must be either 1 or 2")

    return -h


def return_alignment(inputfile: str | Path, stype: int) -> Alignment:
    """Read alignment, remove inserts and convert symbols into integers."""

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
                converted.append(letter2number(char.upper(), stype))
        filtered.append(converted)

    array = np.asarray(filtered, dtype=int)
    q = int(array.max(initial=1))
    length = array.shape[1]
    size = array.shape[0]
    return Alignment(length=length, size=size, alphabet=q, sequences=array)


def _read_fasta_sequences(path: str | Path) -> List[str]:
    """Return sequences contained in a FASTA file as uppercase strings."""

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


def letter2number(char: str, stype: int) -> int:
    """Map amino-acid / nucleotide characters to integer indices.

    This function is deliberately faithful to the MATLAB counterpart.  Unknown
    characters default to the gap state.
    """

    if stype == 1:  # protein alphabet (20 AA + gap)
        amino_map = {
            "-": 1,
            "A": 2,
            "C": 3,
            "D": 4,
            "E": 5,
            "F": 6,
            "G": 7,
            "H": 8,
            "I": 9,
            "K": 10,
            "L": 11,
            "M": 12,
            "N": 13,
            "P": 14,
            "Q": 15,
            "R": 16,
            "S": 17,
            "T": 18,
            "V": 19,
            "W": 20,
            "Y": 21,
        }
        return amino_map.get(char, 1)

    if stype == 2:  # nucleic acids
        nucleotide_map = {
            "A": 1,
            "C": 2,
            "G": 3,
            "T": 4,
            "U": 4,
            "-": 5,
        }
        return nucleotide_map.get(char, 1)

    raise ValueError("stype must be either 1 (protein) or 2 (RNA/DNA)")


__all__ = [
    "Alignment",
    "fastahamiltonian",
    "letter2number",
    "return_alignment",
]

