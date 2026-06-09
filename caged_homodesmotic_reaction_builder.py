#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Caged Molecule Homodesmotic Reaction Builder plugin for MoleditPy.

The plugin detects common local bonding environments in the current molecule and
builds a draft reference-molecule balance for homodesmotic or isodesmic
correction workflows.
"""

from __future__ import annotations

import csv
import html
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised only in hosts without RDKit
    Chem = None  # type: ignore[assignment]

try:
    import numpy as np
    from scipy.optimize import LinearConstraint, milp
except ImportError:
    np = None
    LinearConstraint = None
    milp = None

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import (
        QDialog,
        QFileDialog,
        QHeaderView,
        QHBoxLayout,
        QLabel,
        QMessageBox,
        QPushButton,
        QTableWidget,
        QTableWidgetItem,
        QTextEdit,
        QVBoxLayout,
    )
    from PyQt6.QtGui import QColor, QBrush
except ImportError:  # pragma: no cover - exercised only in non-GUI test hosts
    Qt = None  # type: ignore[assignment]
    QDialog = None  # type: ignore[assignment]
    QFileDialog = None  # type: ignore[assignment]
    QHeaderView = None  # type: ignore[assignment]
    QHBoxLayout = None  # type: ignore[assignment]
    QLabel = None  # type: ignore[assignment]
    QMessageBox = None  # type: ignore[assignment]
    QPushButton = None  # type: ignore[assignment]
    QTableWidget = None  # type: ignore[assignment]
    QTableWidgetItem = None  # type: ignore[assignment]
    QTextEdit = None  # type: ignore[assignment]
    QVBoxLayout = None  # type: ignore[assignment]


PLUGIN_NAME = "Caged Molecule Homodesmotic Reaction Builder"
PLUGIN_VERSION = "0.1.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DEPENDENCIES = ("numpy", "scipy")
PLUGIN_DESCRIPTION = (
    "Detect cage-molecule bonding environments and build a draft homodesmotic "
    "reaction balance."
)

WINDOW_ID = "caged_homodesmotic_reaction_builder"


@dataclass(frozen=True)
class EnvironmentRule:
    name: str
    smarts: str
    reference_smiles: str
    description: str


@dataclass(frozen=True)
class BalanceSpecies:
    name: str
    smiles: str
    description: str


ENVIRONMENTS: tuple[EnvironmentRule, ...] = (
    EnvironmentRule(
        "primary carbon - primary carbon",
        "[CX4H3]-[CX4H3]",
        "CC",
        "Ethane-like terminal alkyl bond.",
    ),
    EnvironmentRule(
        "primary carbon - secondary carbon",
        "[CX4H3]-[CX4H2]",
        "CCC",
        "Propane-like terminal-to-methylene alkyl bond.",
    ),
    EnvironmentRule(
        "primary carbon - tertiary carbon",
        "[CX4H3]-[CX4H1]",
        "CC(C)C",
        "Isobutane-like terminal-to-methine alkyl bond.",
    ),
    EnvironmentRule(
        "primary carbon - quaternary carbon",
        "[CX4H3]-[CX4H0]",
        "CC(C)(C)C",
        "Neopentane-like terminal-to-quaternary alkyl bond.",
    ),
    EnvironmentRule(
        "secondary carbon - secondary carbon",
        "[CX4H2]-[CX4H2]",
        "CCCC",
        "Butane-like methylene-to-methylene alkyl bond.",
    ),
    EnvironmentRule(
        "secondary carbon - tertiary carbon",
        "[CX4H2]-[CX4H1]",
        "CCC(C)C",
        "Branched alkane methylene-to-methine bond.",
    ),
    EnvironmentRule(
        "tertiary carbon - tertiary carbon",
        "[CX4H1]-[CX4H1]",
        "CC(C)C(C)C",
        "Branched alkane methine-to-methine bond.",
    ),
    EnvironmentRule(
        "primary carbon - ether oxygen - primary carbon",
        "[CX4H3]-[OX2H0]-[CX4H3]",
        "COC",
        "Ether oxygen bonded to two primary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - ether oxygen - secondary carbon",
        "[CX4H3]-[OX2H0]-[CX4H2]",
        "COCC",
        "Ether oxygen bonded to primary and secondary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - ether oxygen - tertiary carbon",
        "[CX4H3]-[OX2H0]-[CX4H1]",
        "COC(C)C",
        "Ether oxygen bonded to primary and tertiary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - ether oxygen - quaternary carbon",
        "[CX4H3]-[OX2H0]-[CX4H0]",
        "COC(C)(C)C",
        "Ether oxygen bonded to primary and quaternary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - ether oxygen - secondary carbon",
        "[CX4H2]-[OX2H0]-[CX4H2]",
        "CCOCC",
        "Ether oxygen bonded to two secondary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - ether oxygen - tertiary carbon",
        "[CX4H2]-[OX2H0]-[CX4H1]",
        "CCOC(C)C",
        "Ether oxygen bonded to secondary and tertiary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - ether oxygen - quaternary carbon",
        "[CX4H2]-[OX2H0]-[CX4H0]",
        "CCOC(C)(C)C",
        "Ether oxygen bonded to secondary and quaternary carbons.",
    ),
    EnvironmentRule(
        "tertiary carbon - ether oxygen - tertiary carbon",
        "[CX4H1]-[OX2H0]-[CX4H1]",
        "CC(C)OC(C)C",
        "Ether oxygen bonded to two tertiary carbons.",
    ),
    EnvironmentRule(
        "tertiary carbon - ether oxygen - quaternary carbon",
        "[CX4H1]-[OX2H0]-[CX4H0]",
        "CC(C)OC(C)(C)C",
        "Ether oxygen bonded to tertiary and quaternary carbons.",
    ),
    EnvironmentRule(
        "quaternary carbon - ether oxygen - quaternary carbon",
        "[CX4H0]-[OX2H0]-[CX4H0]",
        "CC(C)(C)OC(C)(C)C",
        "Ether oxygen bonded to two quaternary carbons.",
    ),
    
    # Carbonyls
    EnvironmentRule("Primary-Carbonyl", "[CX4H3]-[CX3](=[OX1])", "CC(=O)C", "Carbonyl carbon bonded to primary carbons."),
    EnvironmentRule("Secondary-Carbonyl", "[CX4H2]-[CX3](=[OX1])", "CCC(=O)C", "Carbonyl carbon bonded to a secondary carbon."),
    EnvironmentRule("Tertiary-Carbonyl", "[CX4H1]-[CX3](=[OX1])", "CC(C)C(=O)C", "Carbonyl carbon bonded to a tertiary carbon."),
    EnvironmentRule("Quaternary-Carbonyl", "[CX4H0]-[CX3](=[OX1])", "CC(C)(C)C(=O)C", "Carbonyl carbon bonded to a quaternary carbon."),
    
    # Amines
    EnvironmentRule("Primary-Primary-Amine", "[CX4H3]-[NX3H2]", "CN", "Primary amine bonded to primary carbon."),
    EnvironmentRule("Secondary-Primary-Amine", "[CX4H2]-[NX3H2]", "CCN", "Primary amine bonded to secondary carbon."),
    EnvironmentRule("Tertiary-Primary-Amine", "[CX4H1]-[NX3H2]", "CC(C)N", "Primary amine bonded to tertiary carbon."),
    EnvironmentRule("Quaternary-Primary-Amine", "[CX4H0]-[NX3H2]", "CC(C)(C)N", "Primary amine bonded to quaternary carbon."),
    EnvironmentRule("Primary-Secondary-Amine", "[CX4H3]-[NX3H1]", "CNC", "Secondary amine bonded to primary carbon."),
    EnvironmentRule("Secondary-Secondary-Amine", "[CX4H2]-[NX3H1]", "CCNC", "Secondary amine bonded to secondary carbon."),
    EnvironmentRule("Tertiary-Secondary-Amine", "[CX4H1]-[NX3H1]", "CC(C)NC", "Secondary amine bonded to tertiary carbon."),
    EnvironmentRule("Quaternary-Secondary-Amine", "[CX4H0]-[NX3H1]", "CC(C)(C)NC", "Secondary amine bonded to quaternary carbon."),
    EnvironmentRule("Primary-Tertiary-Amine", "[CX4H3]-[NX3H0]", "CN(C)C", "Tertiary amine bonded to primary carbon."),
    EnvironmentRule("Secondary-Tertiary-Amine", "[CX4H2]-[NX3H0]", "CCN(C)C", "Tertiary amine bonded to secondary carbon."),
    EnvironmentRule("Tertiary-Tertiary-Amine", "[CX4H1]-[NX3H0]", "CC(C)N(C)C", "Tertiary amine bonded to tertiary carbon."),
    EnvironmentRule("Quaternary-Tertiary-Amine", "[CX4H0]-[NX3H0]", "CC(C)(C)N(C)C", "Tertiary amine bonded to quaternary carbon."),
)


BALANCE_SPECIES: tuple[BalanceSpecies, ...] = (
    BalanceSpecies("methane", "C", "One-carbon saturated hydrocarbon balance species."),
    BalanceSpecies("ethane", "CC", "Two-carbon saturated hydrocarbon balance species."),
    BalanceSpecies("propane", "CCC", "Three-carbon saturated hydrocarbon balance species."),
    BalanceSpecies("butane", "CCCC", "Four-carbon saturated hydrocarbon balance species."),
    BalanceSpecies("Pentane", "CCCCC", "Five-carbon saturated hydrocarbon"),
    BalanceSpecies("Hexane", "CCCCCC", "Six-carbon saturated hydrocarbon"),
    BalanceSpecies("Dimethyl ether", "COC", "Simplest ether"),
    BalanceSpecies("Ethyl methyl ether", "CCOC", "Asymmetric aliphatic ether"),
    BalanceSpecies("Diethyl ether", "CCOCC", "Common symmetric ether"),
    BalanceSpecies("Isobutane", "CC(C)C", "Branched alkane"),
    BalanceSpecies("Neopentane", "CC(C)(C)C", "Highly branched alkane"),
    BalanceSpecies("Isopropyl methyl ether", "COC(C)C", "Branched ether"),
    BalanceSpecies("tert-Butyl methyl ether", "COC(C)(C)C", "Highly branched ether"),
    BalanceSpecies("Formaldehyde", "C=O", "Simplest carbonyl"),
    BalanceSpecies("Acetaldehyde", "CC=O", "Simplest aldehyde"),
    BalanceSpecies("Acetone", "CC(=O)C", "Simplest ketone"),
    BalanceSpecies("Ammonia", "N", "Simplest amine"),
    BalanceSpecies("Methylamine", "CN", "Primary amine"),
    BalanceSpecies("Dimethylamine", "CNC", "Secondary amine"),
    BalanceSpecies("Trimethylamine", "CN(C)C", "Tertiary amine"),
)


@dataclass(frozen=True)
class EnvironmentMatch:
    name: str
    count: int
    reference_smiles: str
    description: str
    original_atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class ReferenceTerm:
    count: int
    smiles: str
    original_atom_indices: tuple[int, ...]


@dataclass(frozen=True)
class BalanceTerm:
    name: str
    smiles: str
    count: int


@dataclass(frozen=True)
class AnalysisResult:
    target_smiles: str
    target_atoms: Counter[str]
    reference_atoms: Counter[str]
    atom_delta: Counter[str]
    left_balance_terms: tuple[BalanceTerm, ...]
    right_balance_terms: tuple[BalanceTerm, ...]
    unresolved_left_atoms: Counter[str]
    unresolved_right_atoms: Counter[str]
    matches: tuple[EnvironmentMatch, ...]
    equation_text: str
    equation_html: str
    is_elemental_balance: bool


def _require_rdkit() -> None:
    if Chem is None:
        raise RuntimeError("RDKit is not available.")


def atom_counts(mol: Any) -> Counter[str]:
    """Return element counts for an RDKit molecule."""
    counts: Counter[str] = Counter()
    if mol is None:
        return counts
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] += 1
    return counts


def count_groups(mol: Any) -> Counter[str]:
    """Return local environment group counts for an RDKit molecule to ensure hyperhomodesmotic balance."""
    counts: Counter[str] = Counter()
    if mol is None:
        return counts
    mol = Chem.AddHs(mol)
    for atom in mol.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        if atomic_num == 1:
            continue
        if atomic_num == 8:
            if atom.GetDegree() == 1:
                counts["=O"] += 1
            else:
                counts["O(C)(C)"] += 1
        elif atomic_num == 7:
            h_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            counts[f"N(H{h_count})"] += 1
        elif atomic_num == 6:
            h_count = 0
            o_count = 0
            n_count = 0
            carbonyl_o = 0
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetAtomicNum() == 1:
                    h_count += 1
                elif neighbor.GetAtomicNum() == 8:
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        carbonyl_o += 1
                    else:
                        o_count += 1
                elif neighbor.GetAtomicNum() == 7:
                    n_count += 1
            if carbonyl_o:
                counts[f"C(=O)(H{h_count})(O{o_count})(N{n_count})"] += 1
            else:
                counts[f"C(H{h_count})(O{o_count})(N{n_count})"] += 1
    return counts


def explicit_hydrogen_copy(mol: Any) -> Any:
    """Return a copy with explicit hydrogens without mutating the host molecule."""
    _require_rdkit()
    return Chem.AddHs(mol) if mol is not None else None


def unique_substructure_matches(mol: Any, smarts: str) -> set[tuple[int, ...]]:
    """Return canonicalized atom-index tuples for one SMARTS pattern."""
    _require_rdkit()
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None or mol is None:
        return set()
    return {tuple(sorted(match)) for match in mol.GetSubstructMatches(pattern)}


def analyze_molecule(mol: Any) -> AnalysisResult:
    """Analyze a molecule and create a draft homodesmotic equation."""
    _require_rdkit()
    if mol is None or mol.GetNumAtoms() == 0:
        return AnalysisResult(
            "",
            Counter(),
            Counter(),
            Counter(),
            (),
            (),
            Counter(),
            Counter(),
            (),
            "No molecule is loaded.",
            "<p>No molecule is loaded.</p>",
            False,
        )

    target_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    target_atoms = atom_counts(explicit_hydrogen_copy(mol))
    target_groups = count_groups(mol)
    reference_atoms: Counter[str] = Counter()
    reference_groups: Counter[str] = Counter()
    matches: list[EnvironmentMatch] = []
    rhs_terms: list[ReferenceTerm] = []

    for rule in ENVIRONMENTS:
        count = len(unique_substructure_matches(mol, rule.smarts))
        if count == 0:
            continue

        matches.append(
            EnvironmentMatch(
                rule.name,
                count,
                rule.reference_smiles,
                rule.description,
                reference_original_atom_indices(rule),
            )
        )
        rhs_terms.append(
            ReferenceTerm(
                count,
                rule.reference_smiles,
                reference_original_atom_indices(rule),
            )
        )

        ref_mol = Chem.MolFromSmiles(rule.reference_smiles)
        if ref_mol is None:
            continue
        for element, number in atom_counts(explicit_hydrogen_copy(ref_mol)).items():
            reference_atoms[element] += number * count
        for group, number in count_groups(ref_mol).items():
            reference_groups[group] += number * count

    atom_delta: Counter[str] = Counter()
    for element in sorted(set(target_atoms) | set(reference_atoms)):
        delta = reference_atoms[element] - target_atoms[element]
        if delta != 0:
            atom_delta[element] = delta

    group_delta: Counter[str] = Counter()
    for group in sorted(set(target_groups) | set(reference_groups)):
        delta = reference_groups[group] - target_groups[group]
        if delta != 0:
            group_delta[group] = delta

    left_balance_terms, right_balance_terms = (), ()
    unresolved_left_atoms, unresolved_right_atoms = Counter(), Counter()
    milp_success = False

    if milp is not None and np is not None:
        left_balance_terms, right_balance_terms, milp_success = build_hyperhomodesmotic_balance_terms(group_delta)

    if not milp_success:
        left_needed = Counter(
            {element: count for element, count in atom_delta.items() if count > 0}
        )
        right_needed = Counter(
            {element: -count for element, count in atom_delta.items() if count < 0}
        )
        left_balance_terms, unresolved_left_atoms = build_balance_terms(left_needed)
        right_balance_terms, unresolved_right_atoms = build_balance_terms(right_needed)

    equation_text = build_equation_text(
        target_smiles,
        target_atoms,
        reference_atoms,
        atom_delta,
        rhs_terms,
        left_balance_terms,
        right_balance_terms,
        unresolved_left_atoms,
        unresolved_right_atoms,
    )
    equation_html = build_equation_html(
        target_smiles,
        target_atoms,
        reference_atoms,
        atom_delta,
        rhs_terms,
        left_balance_terms,
        right_balance_terms,
        unresolved_left_atoms,
        unresolved_right_atoms,
        not milp_success,
    )
    return AnalysisResult(
        target_smiles,
        target_atoms,
        reference_atoms,
        atom_delta,
        tuple(left_balance_terms),
        tuple(right_balance_terms),
        unresolved_left_atoms,
        unresolved_right_atoms,
        tuple(matches),
        equation_text,
        equation_html,
        not milp_success,
    )


def species_atom_counts(smiles: str) -> Counter[str]:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Counter()
    return atom_counts(explicit_hydrogen_copy(mol))


def reference_original_atom_indices(rule: EnvironmentRule) -> tuple[int, ...]:
    """Return reference atoms that correspond to the original matched environment."""
    _require_rdkit()
    ref_mol = Chem.MolFromSmiles(rule.reference_smiles)
    pattern = Chem.MolFromSmarts(rule.smarts)
    if ref_mol is None or pattern is None:
        return ()

    matches = ref_mol.GetSubstructMatches(pattern)
    if not matches:
        return ()
    return tuple(sorted(matches[0]))


def _counter_fits(candidate: Counter[str], remaining: Counter[str]) -> bool:
    return all(remaining[element] >= count for element, count in candidate.items())


def _subtract_counter(remaining: Counter[str], candidate: Counter[str]) -> Counter[str]:
    next_remaining = Counter(remaining)
    for element, count in candidate.items():
        next_remaining[element] -= count
        if next_remaining[element] == 0:
            del next_remaining[element]
    return next_remaining


def _counter_key(values: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted((element, count) for element, count in values.items() if count > 0))


def _solution_score(
    indexes: tuple[int, ...],
    candidates: tuple[tuple[BalanceSpecies, Counter[str]], ...],
) -> tuple[int, int, int, str]:
    heavy_atoms = sum(
        count
        for index in indexes
        for element, count in candidates[index][1].items()
        if element != "H"
    )
    min_heavy = min(
        sum(count for element, count in candidates[index][1].items() if element != "H")
        for index in indexes
    ) if indexes else 0
    names = ",".join(candidates[index][0].name for index in indexes)
    return (len(indexes), -heavy_atoms, -min_heavy, names)


def _search_balance(
    needed_atoms: Counter[str],
    candidates: tuple[tuple[BalanceSpecies, Counter[str]], ...],
) -> tuple[int, ...] | None:
    """Find the smallest exact multiset of balance species for the atom count."""

    @lru_cache(maxsize=None)
    def search(
        remaining_key: tuple[tuple[str, int], ...],
        start_index: int,
    ) -> tuple[int, ...] | None:
        remaining = Counter(dict(remaining_key))
        if not remaining:
            return ()

        best: tuple[int, ...] | None = None
        for index in range(start_index, len(candidates)):
            _species, counts = candidates[index]
            if not _counter_fits(counts, remaining):
                continue

            next_remaining = _subtract_counter(remaining, counts)
            tail = search(_counter_key(next_remaining), index)
            if tail is None:
                continue

            solution = (index, *tail)
            if best is None or _solution_score(solution, candidates) < _solution_score(
                best, candidates
            ):
                best = solution

        return best

    return search(_counter_key(needed_atoms), 0)


def build_hyperhomodesmotic_balance_terms(
    group_delta: Counter[str],
) -> tuple[tuple[BalanceTerm, ...], tuple[BalanceTerm, ...], bool]:
    """Solve for a perfectly balanced hyperhomodesmotic reaction using MILP."""
    if not group_delta:
        return (), (), True
        
    species_list = []
    for species in BALANCE_SPECIES:
        mol = Chem.MolFromSmiles(species.smiles)
        if mol is not None:
            species_list.append((species, count_groups(mol)))
            
    if not species_list:
        return (), (), False

    group_names = sorted(list(set(k for _, c in species_list for k in c.keys()) | set(group_delta.keys())))
    n_groups = len(group_names)
    n_species = len(species_list)

    A = np.zeros((n_groups, n_species))
    for j, (_, c) in enumerate(species_list):
        for i, name in enumerate(group_names):
            A[i, j] = c[name]

    A_eq = np.hstack((A, -A))
    b_eq = np.array([group_delta.get(name, 0) for name in group_names])

    c = np.ones(2 * n_species)
    for j, (species, _) in enumerate(species_list):
        heavy = sum(1 for char in species.smiles if char.isalpha() and char != 'H')
        cost = 1.0
        if heavy == 1:
            cost = 1.02
        elif heavy == 2:
            cost = 1.01
        c[j] = cost
        c[n_species + j] = cost

    integrality = np.ones(2 * n_species)

    try:
        res = milp(c=c, constraints=LinearConstraint(A_eq, b_eq, b_eq), integrality=integrality)
        if not res.success:
            return (), (), False
    except Exception:
        return (), (), False

    u = np.round(res.x[:n_species]).astype(int)
    v = np.round(res.x[n_species:]).astype(int)
    
    left_terms = []
    for j, count in enumerate(u):
        if count > 0:
            species = species_list[j][0]
            left_terms.append(BalanceTerm(species.name, species.smiles, int(count)))
            
    right_terms = []
    for j, count in enumerate(v):
        if count > 0:
            species = species_list[j][0]
            right_terms.append(BalanceTerm(species.name, species.smiles, int(count)))
            
    return tuple(left_terms), tuple(right_terms), True


def build_balance_terms(
    needed_atoms: Counter[str],
) -> tuple[tuple[BalanceTerm, ...], Counter[str]]:
    """Build simple molecule terms that exactly cover the requested atom counts."""
    if not needed_atoms:
        return (), Counter()

    allowed_elements = set(needed_atoms)
    candidates: list[tuple[BalanceSpecies, Counter[str]]] = []
    for species in BALANCE_SPECIES:
        counts = species_atom_counts(species.smiles)
        if counts and set(counts).issubset(allowed_elements):
            candidates.append((species, counts))

    candidates.sort(
        key=lambda item: (
            -sum(1 for element in item[1] if element != "H"),
            -sum(item[1].values()),
            item[0].name,
        )
    )

    candidate_tuple = tuple(candidates)
    solution = _search_balance(needed_atoms, candidate_tuple)
    if solution is None:
        return (), Counter(needed_atoms)

    grouped = Counter(candidate_tuple[index][0] for index in solution)
    terms = tuple(
        BalanceTerm(species.name, species.smiles, count)
        for species, count in sorted(grouped.items(), key=lambda item: item[0].name)
    )
    return terms, Counter()


def format_counter(values: Counter[str]) -> str:
    if not values:
        return "none"
    return ", ".join(f"{element}: {values[element]}" for element in sorted(values))


def build_equation_text(
    target_smiles: str,
    target_atoms: Counter[str],
    reference_atoms: Counter[str],
    atom_delta: Counter[str],
    rhs_terms: Iterable[ReferenceTerm],
    left_balance_terms: Iterable[BalanceTerm],
    right_balance_terms: Iterable[BalanceTerm],
    unresolved_left_atoms: Counter[str],
    unresolved_right_atoms: Counter[str],
) -> str:
    left_balance = format_terms(left_balance_terms)
    right_balance = format_terms(right_balance_terms)
    unresolved_left = format_counter(unresolved_left_atoms)
    unresolved_right = format_counter(unresolved_right_atoms)
    target_label = target_smiles or "Target"
    lhs_parts = [target_label]
    if left_balance != "none":
        lhs_parts.append(left_balance)
    if unresolved_left != "none":
        lhs_parts.append(f"unresolved left species ({unresolved_left})")
    lhs = " + ".join(lhs_parts)

    rhs_references = format_reference_terms(rhs_terms)
    rhs_parts = [rhs_references]
    if right_balance != "none":
        rhs_parts.append(right_balance)
    if unresolved_right != "none":
        rhs_parts.append(f"unresolved right species ({unresolved_right})")
    rhs = " + ".join(rhs_parts)
    return (
        "Homodesmotic Draft Equation\n"
        "===========================\n\n"
        f"Target atom count: {format_counter(target_atoms)}\n"
        f"Reference-side atom count: {format_counter(reference_atoms)}\n"
        f"Reference minus target atom delta: {format_counter(atom_delta)}\n\n"
        f"Left-side balancing species: {left_balance}\n"
        f"Right-side balancing species: {right_balance}\n"
        f"Unresolved left-side atoms: {unresolved_left}\n"
        f"Unresolved right-side atoms: {unresolved_right}\n\n"
        f"{lhs} -> {rhs}\n\n"
        "Energy expression template\n"
        "--------------------------\n"
        "Strain or stabilization energy can be estimated from a balanced reaction:\n"
        "SE = sum(E_products) - sum(E_reactants)\n\n"
        "Notes\n"
        "-----\n"
        "This tool detects predefined local environments and proposes reference "
        "molecules. Simple balancing species are added automatically when the "
        "atom-count difference can be represented by the built-in molecule library."
    )


def format_terms(terms: Iterable[BalanceTerm]) -> str:
    parts = [
        f"{term.count} {term.smiles} ({term.name})"
        for term in terms
        if term.count > 0
    ]
    return " + ".join(parts) if parts else "none"


def format_reference_terms(terms: Iterable[ReferenceTerm]) -> str:
    parts = [f"{term.count} {term.smiles}" for term in terms if term.count > 0]
    return " + ".join(parts) if parts else "no reference molecules detected"


def _span(text: str, color: str, title: str = "") -> str:
    safe_text = html.escape(text)
    safe_title = html.escape(title)
    title_attr = f' title="{safe_title}"' if safe_title else ""
    return f'<span style="color:{color}; font-weight:600;"{title_attr}>{safe_text}</span>'


def color_smiles_atoms(smiles: str, color: str, title: str) -> str:
    """Color only atom tokens inside a SMILES string."""
    return color_smiles_atoms_by_index(smiles, {}, color, title)


def color_smiles_atoms_by_index(
    smiles: str,
    indexed_colors: dict[int, tuple[str, str]],
    default_color: str,
    default_title: str,
) -> str:
    """Color atom tokens inside a SMILES string using atom-token indexes."""
    organic_two_letter_atoms = {"Cl", "Br"}
    organic_one_letter_atoms = set("BCNOPSFIbcnops")
    pieces: list[str] = []
    index = 0
    atom_index = 0

    while index < len(smiles):
        char = smiles[index]
        if char == "[":
            end = smiles.find("]", index + 1)
            if end != -1:
                color, title = indexed_colors.get(
                    atom_index, (default_color, default_title)
                )
                pieces.append(_span(smiles[index : end + 1], color, title))
                atom_index += 1
                index = end + 1
                continue

        two_letter = smiles[index : index + 2]
        if two_letter in organic_two_letter_atoms:
            color, title = indexed_colors.get(atom_index, (default_color, default_title))
            pieces.append(_span(two_letter, color, title))
            atom_index += 1
            index += 2
            continue

        if char in organic_one_letter_atoms:
            color, title = indexed_colors.get(atom_index, (default_color, default_title))
            pieces.append(_span(char, color, title))
            atom_index += 1
            index += 1
            continue

        pieces.append(html.escape(char))
        index += 1

    return "".join(pieces)


def color_equation_term(term: str, color: str, title: str) -> str:
    """Color atom letters in a coefficient + SMILES equation term."""
    count, separator, smiles = term.partition(" ")
    if separator:
        return f"{html.escape(count)} {color_smiles_atoms(smiles, color, title)}"
    return color_smiles_atoms(term, color, title)


def color_reference_term(
    term: ReferenceTerm,
    original_color: str,
    added_color: str,
) -> str:
    indexed_colors = {
        index: (original_color, "Original cage-environment atom in reference fragment")
        for index in term.original_atom_indices
    }
    smiles = color_smiles_atoms_by_index(
        term.smiles,
        indexed_colors,
        added_color,
        "Added capping atom in reference fragment",
    )
    return f"{term.count} {smiles}"


def _html_counter(values: Counter[str], color: str, title: str) -> str:
    if not values:
        return "none"
    return ", ".join(
        f"{_span(element, color, title)}: {values[element]}"
        for element in sorted(values)
    )


def _html_terms(terms: Iterable[BalanceTerm], color: str, title: str) -> str:
    parts = [
        f'{term.count} {color_smiles_atoms(term.smiles, color, title)} '
        f'<span style="color:#9aa0a6;">({html.escape(term.name)})</span>'
        for term in terms
        if term.count > 0
    ]
    return " + ".join(parts) if parts else "none"


def build_equation_html(
    target_smiles: str,
    target_atoms: Counter[str],
    reference_atoms: Counter[str],
    atom_delta: Counter[str],
    rhs_terms: Iterable[ReferenceTerm],
    left_balance_terms: Iterable[BalanceTerm],
    right_balance_terms: Iterable[BalanceTerm],
    unresolved_left_atoms: Counter[str],
    unresolved_right_atoms: Counter[str],
    is_elemental_balance: bool = False,
) -> str:
    target_color = "#8ab4f8"
    reference_color = "#8ab4f8"
    left_added_color = "#fdd663"
    right_added_color = "#c58af9"
    unresolved_color = "#f28b82"

    left_balance = _html_terms(left_balance_terms, left_added_color, "Added left-side balance species")
    right_balance = _html_terms(right_balance_terms, left_added_color, "Added right-side balance species")
    unresolved_left = _html_counter(unresolved_left_atoms, unresolved_color, "Unresolved left-side atom")
    unresolved_right = _html_counter(unresolved_right_atoms, unresolved_color, "Unresolved right-side atom")

    target_label = target_smiles or "Target"
    lhs_parts = [
        color_smiles_atoms(target_label, target_color, "Original target atom")
    ]
    if left_balance != "none":
        lhs_parts.append(left_balance)
    if unresolved_left != "none":
        lhs_parts.append(f'{_span("unresolved left species", unresolved_color)} ({unresolved_left})')

    rhs_parts = [
        " + ".join(
            color_reference_term(term, reference_color, left_added_color)
            for term in rhs_terms
        )
        or "no reference molecules detected"
    ]
    if right_balance != "none":
        rhs_parts.append(right_balance)
    if unresolved_right != "none":
        rhs_parts.append(f'{_span("unresolved right species", unresolved_color)} ({unresolved_right})')

    warning_html = ""
    if is_elemental_balance:
        warning_html = '<p style="color:#f28b82; font-weight:bold;">⚠️ Note: Calculated in elemental balance mode due to environment matching constraints.</p>'

    return (
        '<div style="font-family:Consolas, monospace; color:#e8eaed;">'
        '<h3 style="margin:0 0 8px 0; color:#e8eaed;">Homodesmotic Draft Equation</h3>'
        f'{warning_html}'
        f'<p><span style="color:#9aa0a6;">Target atom count:</span> '
        f'{_html_counter(target_atoms, target_color, "Original target atom")}</p>'
        f'<p><span style="color:#9aa0a6;">Reference-side atom count:</span> '
        f'{_html_counter(reference_atoms, reference_color, "Detected reference atom")}</p>'
        f'<p><span style="color:#9aa0a6;">Reference minus target atom delta:</span> '
        f'{_html_counter(atom_delta, "#e8eaed", "Atom-count difference")}</p>'
        f'<p><span style="color:#9aa0a6;">Left-side balancing species:</span> {left_balance}</p>'
        f'<p><span style="color:#9aa0a6;">Right-side balancing species:</span> {right_balance}</p>'
        f'<p><span style="color:#9aa0a6;">Unresolved left-side atoms:</span> {unresolved_left}</p>'
        f'<p><span style="color:#9aa0a6;">Unresolved right-side atoms:</span> {unresolved_right}</p>'
        '<hr style="border:0; border-top:1px solid #3c4043;">'
        f'<p style="font-size:14px;">{" + ".join(lhs_parts)} '
        f'<span style="color:#e8eaed;">-&gt;</span> {" + ".join(rhs_parts)}</p>'
        '<p style="color:#9aa0a6;">Blue = original target and reference cores, '
        'yellow = automatically added balance species and caps, red = unresolved.</p>'
        '</div>'
    )


def _status(context: Any, message: str, timeout: int = 3000) -> None:
    if hasattr(context, "show_status_message"):
        context.show_status_message(message, timeout)


def _current_molecule(context: Any) -> Any:
    main_window = context.get_main_window() if hasattr(context, "get_main_window") else None
    state_manager = getattr(main_window, "state_manager", None)
    data = getattr(state_manager, "data", None)
    if hasattr(data, "to_rdkit_mol"):
        try:
            mol = data.to_rdkit_mol()
            if mol is not None:
                return mol
        except Exception:
            pass
    return getattr(context, "current_molecule", None)


def _load_smiles_with_host(context: Any, smiles: str) -> bool:
    """Load a SMILES string through MoleditPy's native importer when available."""
    main_window = context.get_main_window() if hasattr(context, "get_main_window") else None
    importer = getattr(main_window, "string_importer_manager", None)
    load_from_smiles = getattr(importer, "load_from_smiles", None)
    if not callable(load_from_smiles):
        return False

    load_from_smiles(smiles)
    return True


if QDialog is not None:

    class HomodesmoticAnalyzerDialog(QDialog):
        """Qt dialog for the analyzer."""

        def __init__(self, context: Any) -> None:
            super().__init__(parent=context.get_main_window())
            self.context = context
            self.last_result = AnalysisResult(
                "",
                Counter(),
                Counter(),
                Counter(),
                (),
                (),
                Counter(),
                Counter(),
                (),
                "",
                "",
            )

            self.setWindowTitle("Caged Molecule Homodesmotic Reaction Builder")
            self.resize(820, 640)

            layout = QVBoxLayout(self)

            intro = QLabel(
                "Detected environments are mapped to small reference molecules. "
                "Use the atom-balance summary to complete the final reaction."
            )
            intro.setWordWrap(True)
            layout.addWidget(intro)

            self.warning_label = QLabel(
                "⚠️ SciPy not detected or exact MILP balance impossible; operating in elemental balance mode."
            )
            self.warning_label.setStyleSheet("color: #f28b82; font-weight: bold;")
            self.warning_label.setVisible(False)
            layout.addWidget(self.warning_label)

            self.table = QTableWidget(0, 4)
            self.table.setHorizontalHeaderLabels(
                ["Environment", "Count", "Reference SMILES", "Action"]
            )
            self.table.horizontalHeader().setSectionResizeMode(
                0, QHeaderView.ResizeMode.Stretch
            )
            self.table.horizontalHeader().setSectionResizeMode(
                2, QHeaderView.ResizeMode.ResizeToContents
            )
            self.table.verticalHeader().setVisible(False)
            layout.addWidget(self.table, 1)

            self.equation_box = QTextEdit()
            self.equation_box.setReadOnly(True)
            self.equation_box.setStyleSheet(
                "background-color: #202124; color: #e8eaed; font-family: Consolas, monospace;"
            )
            self.equation_box.setMinimumHeight(190)
            layout.addWidget(self.equation_box)

            button_layout = QHBoxLayout()
            self.analyze_button = QPushButton("Analyze Current Molecule")
            self.export_button = QPushButton("Export Analysis")
            button_layout.addWidget(self.analyze_button)
            button_layout.addStretch()
            button_layout.addWidget(self.export_button)
            layout.addLayout(button_layout)

            self.analyze_button.clicked.connect(self.refresh_analysis)
            self.export_button.clicked.connect(self.export_analysis)

            self.refresh_analysis()

        def refresh_analysis(self) -> None:
            if Chem is None:
                _status(self.context, "RDKit is not available.", 5000)
                return

            mol = _current_molecule(self.context)
            self.last_result = analyze_molecule(mol)
            self._populate_table(self.last_result)
            self.equation_box.setHtml(self.last_result.equation_html)
            self.warning_label.setVisible(self.last_result.is_elemental_balance)

            if mol is None or mol.GetNumAtoms() == 0:
                _status(self.context, "No molecule is loaded.", 3000)
            else:
                _status(
                    self.context,
                    f"Detected {len(self.last_result.matches)} environment types.",
                    3000,
                )

        def _populate_table(self, result: AnalysisResult) -> None:
            matches = result.matches
            left = result.left_balance_terms
            right = result.right_balance_terms
            
            self.table.setRowCount(len(matches) + len(left) + len(right))
            
            ref_brush = QBrush(QColor("#1967d2"))
            balance_brush = QBrush(QColor("#b06000"))
            
            row = 0
            for match in matches:
                self._set_colored_row(row, f"Ref: {match.name}", str(match.count), match.reference_smiles, ref_brush)
                self._add_load_button(row, match.reference_smiles, match.description)
                row += 1
                
            for term in left:
                self._set_colored_row(row, f"Left Balance: {term.name}", str(term.count), term.smiles, balance_brush)
                self._add_load_button(row, term.smiles, "Added left-side balance species")
                row += 1
                
            for term in right:
                self._set_colored_row(row, f"Right Balance: {term.name}", str(term.count), term.smiles, balance_brush)
                self._add_load_button(row, term.smiles, "Added right-side balance species")
                row += 1

        def _set_colored_row(self, row: int, col0: str, col1: str, col2: str, brush: QBrush) -> None:
            for col, text in enumerate((col0, col1, col2)):
                item = QTableWidgetItem(text)
                item.setForeground(brush)
                self.table.setItem(row, col, item)

        def _add_load_button(self, row: int, smiles: str, tooltip: str) -> None:
            load_button = QPushButton("Load Species")
            load_button.setToolTip(tooltip)
            load_button.clicked.connect(
                lambda _checked=False, s=smiles: self.load_reference(s)
            )
            self.table.setCellWidget(row, 3, load_button)

        def load_reference(self, smiles: str) -> None:
            if _load_smiles_with_host(self.context, smiles):
                _status(self.context, f"Loaded reference from SMILES: {smiles}", 3000)
                return

            QMessageBox.information(
                self,
                "SMILES Importer Unavailable",
                "The MoleditPy SMILES importer is not available in this context.",
            )
            _status(self.context, "SMILES importer is not available.", 5000)

        def export_analysis(self) -> None:
            path, _ = QFileDialog.getSaveFileName(
                self,
                "Export Homodesmotic Reaction Draft",
                "caged_homodesmotic_reaction_draft.csv",
                "CSV Files (*.csv);;HTML Files (*.html);;Text Files (*.txt)",
            )
            if not path:
                return

            try:
                export_analysis(path, self.last_result)
            except OSError as exc:
                QMessageBox.critical(self, "Export Failed", str(exc))
                _status(self.context, f"Export failed: {exc}", 5000)
                return

            _status(self.context, f"Analysis exported to {path}", 3000)


def export_analysis(path: str | Path, result: AnalysisResult) -> None:
    """Export analysis output as CSV or plain text."""
    output_path = Path(path)
    if output_path.suffix.lower() in {".html", ".htm"}:
        output_path.write_text(
            "<!doctype html>\n"
            "<html><head><meta charset=\"utf-8\">"
            "<title>Caged Molecule Homodesmotic Reaction Builder</title>"
            "<style>body{background:#202124;color:#e8eaed;margin:24px;}"
            "table{border-collapse:collapse;margin-top:20px;}"
            "th,td{border:1px solid #3c4043;padding:6px 10px;}"
            "th{color:#e8eaed;background:#303134;}</style>"
            "</head><body>\n"
            f"{result.equation_html}\n"
            "<table><thead><tr><th>Environment</th><th>Count</th>"
            "<th>Reference SMILES</th><th>Description</th></tr></thead><tbody>\n"
            + "\n".join(
                "<tr>"
                f"<td>{html.escape(match.name)}</td>"
                f"<td>{match.count}</td>"
                f"<td>{html.escape(match.reference_smiles)}</td>"
                f"<td>{html.escape(match.description)}</td>"
                "</tr>"
                for match in result.matches
            )
            + "\n</tbody></table>\n"
            "</body></html>\n",
            encoding="utf-8",
        )
        return

    if output_path.suffix.lower() == ".txt":
        output_path.write_text(result.equation_text + "\n", encoding="utf-8")
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Environment", "Count", "Reference SMILES", "Description"])
        for match in result.matches:
            writer.writerow(
                [match.name, match.count, match.reference_smiles, match.description]
            )
        writer.writerow([])
        writer.writerow(["Target atom count", format_counter(result.target_atoms)])
        writer.writerow(["Reference-side atom count", format_counter(result.reference_atoms)])
        writer.writerow(["Reference minus target atom delta", format_counter(result.atom_delta)])
        writer.writerow(["Left-side balancing species", format_terms(result.left_balance_terms)])
        writer.writerow(["Right-side balancing species", format_terms(result.right_balance_terms)])
        writer.writerow(["Unresolved left-side atoms", format_counter(result.unresolved_left_atoms)])
        writer.writerow(["Unresolved right-side atoms", format_counter(result.unresolved_right_atoms)])
        writer.writerow([])
        writer.writerow(["Equation"])
        writer.writerow([result.equation_text])


def open_analyzer_dialog(context: Any) -> None:
    if QDialog is None:
        _status(context, "PyQt6 is not available.", 5000)
        return
    if Chem is None:
        _status(context, "RDKit is not available.", 5000)
        return

    window = context.get_window(WINDOW_ID)
    if window:
        window.show()
        window.raise_()
        window.activateWindow()
        if hasattr(window, "refresh_analysis"):
            window.refresh_analysis()
        return

    dialog = HomodesmoticAnalyzerDialog(context)
    context.register_window(WINDOW_ID, dialog)
    dialog.show()


def initialize(context: Any) -> None:
    """Initialize the plugin inside MoleditPy."""
    context.add_analysis_tool(PLUGIN_NAME, lambda: open_analyzer_dialog(context))
    _status(context, f"{PLUGIN_NAME} loaded.", 3000)
