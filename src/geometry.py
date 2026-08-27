"""Cross-chain interface geometry feature computation.

Computes a 22-dimensional geometry descriptor for each mutation site,
capturing its spatial relationship to the partner chain in a protein complex.

Features are grouped into three categories:
  (i)   Cross-chain distance features (9d)
  (ii)  Cross-chain contact count features (9d)
  (iii) Contact residue chemical type features (4d)

All features are computed directly from PDB heavy-atom coordinates.
No learned parameters are involved — this is a purely structural descriptor.
"""

import math
from pathlib import Path

import numpy as np

# Standard amino acid chemical classes
AA_HYDROPHOBIC = set("AILMFWVY")
AA_POLAR = set("STNQCY")
AA_CHARGED = set("DEKRH")
AA_AROMATIC = set("FWYH")

# Three-letter to one-letter amino acid mapping
AA3_TO_1 = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def compute_site_geometry(
    site_atoms: np.ndarray,
    partner_residues: list[tuple[str, np.ndarray]],
) -> np.ndarray:
    """Compute 22d cross-chain geometry descriptor for one mutation site.

    Args:
        site_atoms: Heavy-atom coordinates of the mutation site, shape (N_atoms, 3).
        partner_residues: List of (amino_acid_1letter, coords) for each partner
            chain residue, where coords is shape (M_atoms, 3).

    Returns:
        22-dimensional feature vector (float32).
    """
    if len(site_atoms) == 0 or len(partner_residues) == 0:
        return np.zeros(22, dtype=np.float32)

    a = np.asarray(site_atoms, dtype=np.float32)

    # Compute minimum distance from site to each partner residue
    residue_dists = []
    atom_pair_counts = {4.0: 0, 5.0: 0, 6.0: 0}
    contact_class_counts = {"hydrophobic": 0, "polar": 0, "charged": 0, "aromatic": 0}

    for aa, coords in partner_residues:
        b = np.asarray(coords, dtype=np.float32)
        # Pairwise distance matrix between site atoms and partner atoms
        d = np.sqrt(np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2))
        min_d = float(d.min())
        residue_dists.append(min_d)

        # Atom-pair contact counts at multiple thresholds
        for cutoff in atom_pair_counts:
            atom_pair_counts[cutoff] += int((d <= cutoff).sum())

        # Chemical class of contacted residues (within 5 Å)
        if min_d <= 5.0:
            contact_class_counts["hydrophobic"] += int(aa in AA_HYDROPHOBIC)
            contact_class_counts["polar"] += int(aa in AA_POLAR)
            contact_class_counts["charged"] += int(aa in AA_CHARGED)
            contact_class_counts["aromatic"] += int(aa in AA_AROMATIC)

    dists = np.asarray(residue_dists, dtype=np.float32)
    sorted_d = np.sort(dists)
    min_d = float(sorted_d[0])
    top5 = sorted_d[: min(5, len(sorted_d))]

    # Residue-level contact counts at multiple distance shells
    counts = {c: int((dists <= c).sum()) for c in (4.0, 5.0, 6.0, 8.0, 10.0)}
    denom10 = max(1, counts[10.0])

    # === Assemble 22d feature vector ===
    features = [
        # (i) Cross-chain distance features (9d)
        min_d,                                      # minimum distance to partner
        math.log1p(min_d),                          # log-scale minimum distance
        1.0 / max(min_d, 1e-3),                     # inverse distance (proximity)
        math.exp(-min_d / 2.0),                     # exponential decay proximity
        float(top5.mean()),                         # mean of 5 nearest residue distances
        float(top5.std()) if len(top5) > 1 else 0.0,  # std of nearest distances
        float(np.median(sorted_d)),                 # median distance to all partners
        float(np.quantile(sorted_d, 0.25)),         # 25th percentile
        float(np.quantile(sorted_d, 0.75)),         # 75th percentile
        # (ii) Cross-chain contact count features (9d)
        math.log1p(counts[4.0]),                    # residues within 4 Å (log)
        math.log1p(counts[5.0]),                    # residues within 5 Å
        math.log1p(counts[6.0]),                    # residues within 6 Å
        math.log1p(counts[8.0]),                    # residues within 8 Å
        math.log1p(counts[10.0]),                   # residues within 10 Å
        counts[5.0] / float(denom10),               # contact density (5Å / 10Å)
        math.log1p(atom_pair_counts[4.0]),          # atom pairs within 4 Å
        math.log1p(atom_pair_counts[5.0]),          # atom pairs within 5 Å
        math.log1p(atom_pair_counts[6.0]),          # atom pairs within 6 Å
        # (iii) Contact residue chemical type features (4d)
        math.log1p(len(site_atoms)),                # site size (normalization)
        math.log1p(len(partner_residues)),           # partner size
        contact_class_counts["hydrophobic"] / float(max(1, counts[5.0])),  # hydrophobic fraction
        contact_class_counts["charged"] / float(max(1, counts[5.0])),      # charged fraction
    ]

    return np.asarray(features, dtype=np.float32)


def compute_geometry_from_pdb(
    pdb_path: str,
    chain: str,
    position: str,
    partner_chains: list[str],
) -> np.ndarray:
    """Compute geometry features for a mutation site from a PDB file.

    Args:
        pdb_path: Path to the PDB structure file.
        chain: Chain ID of the mutation site.
        position: Residue number (PDB numbering) of the mutation site.
        partner_chains: Chain IDs of the binding partner.

    Returns:
        22d geometry feature vector.
    """
    from Bio.PDB import PDBParser

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("complex", pdb_path)
    model = structure[0]

    # Get mutation site atoms
    site_atoms = []
    for residue in model[chain].get_residues():
        resnum = str(residue.get_id()[1])
        icode = residue.get_id()[2].strip()
        res_pos = f"{resnum}{icode}" if icode else resnum
        if res_pos == position:
            for atom in residue.get_atoms():
                if atom.element != "H":
                    site_atoms.append(atom.get_vector().get_array())
            break

    if not site_atoms:
        return np.zeros(22, dtype=np.float32)

    # Get partner chain residues
    partner_residues = []
    for pchain_id in partner_chains:
        if pchain_id not in model:
            continue
        for residue in model[pchain_id].get_residues():
            resname = residue.get_resname().strip()
            aa = AA3_TO_1.get(resname, "X")
            atoms = []
            for atom in residue.get_atoms():
                if atom.element != "H":
                    atoms.append(atom.get_vector().get_array())
            if atoms:
                partner_residues.append((aa, np.array(atoms)))

    return compute_site_geometry(np.array(site_atoms), partner_residues)
