"""Minimal example showing how to run the Python DCA utilities."""

from pathlib import Path

from fastahamiltonian import fastahamiltonian
from master_dca_parameters_length import master_dca_parameters_length

DATA = Path(__file__).with_name("example_alignment.fasta")


def main() -> None:
    # Estimate couplings and fields from the example alignment.
    couplings, local_fields = master_dca_parameters_length(DATA, length=20)
    print("Couplings shape:", couplings.shape)
    print("Local fields shape:", local_fields.shape)

    # Evaluate Hamiltonians for all sequences in the FASTA file.
    energies = fastahamiltonian(DATA, couplings, local_fields, htype=2, n1=0, stype=2)
    print("Hamiltonians:")
    for idx, value in enumerate(energies, start=1):
        print(f"  sequence {idx}: {value:.6f}")


if __name__ == "__main__":
    main()
