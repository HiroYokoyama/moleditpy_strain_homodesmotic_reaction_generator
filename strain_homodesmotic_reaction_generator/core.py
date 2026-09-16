#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Core computational logic for Strain Homodesmotic Reaction Generator.
"""

from __future__ import annotations

import csv
import html
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover
    Chem = None  # type: ignore[assignment]

try:
    import numpy as np
    from scipy.optimize import Bounds, LinearConstraint, milp
except ImportError:
    np = None
    Bounds = None
    LinearConstraint = None
    milp = None


from .data import (
    EnvironmentRule,
    BalanceSpecies,
    UserSpecies,
    SIDE_LEFT,
    SIDE_RIGHT,
    ENVIRONMENTS,
    BALANCE_SPECIES,
    _BALANCE_CORE_MAP,
)


@dataclass(frozen=True)
class EnvironmentMatch:
    name: str
    count: int
    reference_smiles: str
    description: str
    original_atom_indices: tuple[int, ...]
    user_defined: bool = False


@dataclass(frozen=True)
class ReferenceTerm:
    count: int
    smiles: str
    original_atom_indices: tuple[int, ...]
    user_defined: bool = False


@dataclass(frozen=True)
class BalanceTerm:
    name: str
    smiles: str
    count: int
    description: str = ""
    user_defined: bool = False


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
    target_bonds: Counter[str]
    reaction_bonds_delta: Counter[str]
    reaction_type: str
    lhs_bonds: Counter[str]
    rhs_bonds: Counter[str]
    reference_overrides: tuple[tuple[str, str], ...] = ()
    user_species: tuple[UserSpecies, ...] = ()
    unmet_required: tuple[str, ...] = ()
    cancelled_references: tuple[str, ...] = ()
    excluded_species: tuple[str, ...] = ()
    ignored_exclusions: tuple[str, ...] = ()


#: Shown wherever the tool states a verdict. It conserves what it counts, and
#: what it counts is not the whole of the chemistry.
SELF_CHECK_NOTE = (
    "This is a generated draft. Check the equation and the reference molecules "
    "yourself before using the energies."
)


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
            if atom.GetIsAromatic():
                counts["O_ar"] += 1
            elif atom.GetDegree() == 1:
                counts["=O"] += 1
            else:
                counts["O(C)(C)"] += 1
        elif atomic_num == 7:
            h_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            if atom.GetIsAromatic():
                counts[f"N_ar(H{h_count})"] += 1
            else:
                counts[f"N(H{h_count})"] += 1
        elif atomic_num == 6:
            h_count = 0
            o_single = 0
            o_double = 0
            n_single = 0
            n_double = 0
            n_triple = 0
            c_double = 0
            c_triple = 0
            c_aromatic = 0
            for bond in atom.GetBonds():
                neighbor = bond.GetOtherAtom(atom)
                if neighbor.GetAtomicNum() == 1:
                    h_count += 1
                elif neighbor.GetAtomicNum() == 8:
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        o_double += 1
                    else:
                        o_single += 1
                elif neighbor.GetAtomicNum() == 7:
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        n_double += 1
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        n_triple += 1
                    else:
                        n_single += 1
                elif neighbor.GetAtomicNum() == 6:
                    if bond.GetBondType() == Chem.BondType.DOUBLE:
                        c_double += 1
                    elif bond.GetBondType() == Chem.BondType.TRIPLE:
                        c_triple += 1
                    elif bond.GetBondType() == Chem.BondType.AROMATIC:
                        c_aromatic += 1

            # Heteroatoms outside C/N/O used to leave no trace here, so a
            # carbon bonded to Si, S, P or a halogen was indistinguishable
            # from one bonded to nothing. Appended only when present, so keys
            # for pure C/H/N/O molecules are unchanged.
            other = _other_heteroatom_suffix(atom)
            if atom.GetIsAromatic():
                counts[
                    f"C_ar(H{h_count})(O_s{o_single})(O_d{o_double})(N_s{n_single})(N_d{n_double})(N_t{n_triple})(ar{c_aromatic}){other}"
                ] += 1
            else:
                counts[
                    f"C(H{h_count})(O_s{o_single})(O_d{o_double})(N_s{n_single})(N_d{n_double})(N_t{n_triple})(=C{c_double})(#C{c_triple}){other}"
                ] += 1
        else:
            # Any other heavy atom (S, P, Si, B, Se, halogens, ...). Without
            # this branch such atoms contributed no group at all, so the MILP
            # could not see them: it balanced cyclopropane by adding
            # tetramethylsilane, whose Si and carbons were both invisible.
            h_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
            neighbours = _neighbour_signature(atom)
            prefix = (
                f"{atom.GetSymbol()}_ar" if atom.GetIsAromatic() else atom.GetSymbol()
            )
            counts[f"{prefix}(H{h_count}){neighbours}"] += 1
    return counts


#: Elements whose bonding is spelled out explicitly in the carbon descriptor.
_DESCRIBED_ELEMENTS = frozenset({1, 6, 7, 8})


def _bond_symbol(bond: Any) -> str:
    if bond.GetBondType() == Chem.BondType.DOUBLE:
        return "="
    if bond.GetBondType() == Chem.BondType.TRIPLE:
        return "#"
    if bond.GetBondType() == Chem.BondType.AROMATIC:
        return ":"
    return "-"


def _neighbour_signature(atom: Any) -> str:
    """Sorted (bond order, element) list of an atom's heavy neighbours."""
    parts = sorted(
        f"{_bond_symbol(bond)}{bond.GetOtherAtom(atom).GetSymbol()}"
        for bond in atom.GetBonds()
        if bond.GetOtherAtom(atom).GetAtomicNum() != 1
    )
    return "(" + ",".join(parts) + ")" if parts else "()"


def _other_heteroatom_suffix(atom: Any) -> str:
    """Descriptor for neighbours outside C/H/N/O; empty when there are none."""
    parts = sorted(
        f"{_bond_symbol(bond)}{bond.GetOtherAtom(atom).GetSymbol()}"
        for bond in atom.GetBonds()
        if bond.GetOtherAtom(atom).GetAtomicNum() not in _DESCRIBED_ELEMENTS
    )
    return "(X:" + ",".join(parts) + ")" if parts else ""


def get_atom_state(atom: Any) -> str:
    """Return a string representation of an atom's hybridization and hydrogen count."""
    symbol = atom.GetSymbol()
    if symbol == "H":
        return "H"
    hyb = str(atom.GetHybridization()).lower()
    h_count = sum(1 for n in atom.GetNeighbors() if n.GetAtomicNum() == 1)
    return f"{symbol}({hyb},H{h_count})"


def bond_counts(mol: Any) -> Counter[str]:
    """Return bond type counts for an RDKit molecule."""
    counts: Counter[str] = Counter()
    if mol is None:
        return counts
    mol_with_hs = explicit_hydrogen_copy(mol)
    return bond_counts_helper(mol_with_hs)


def simple_bond_counts_helper(mol: Any) -> Counter[str]:
    """Return simple element-pair bond type counts (without hybridization)."""
    counts: Counter[str] = Counter()
    if mol is None:
        return counts
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetSymbol()
        a2 = bond.GetEndAtom().GetSymbol()
        pair = sorted([a1, a2])
        btype = bond.GetBondType()
        if btype == Chem.BondType.SINGLE:
            suffix = " (single)"
        elif btype == Chem.BondType.DOUBLE:
            suffix = " (double)"
        elif btype == Chem.BondType.TRIPLE:
            suffix = " (triple)"
        elif btype == Chem.BondType.AROMATIC:
            suffix = " (aromatic)"
        else:
            suffix = f" ({str(btype).lower()})"
        counts[f"{pair[0]}-{pair[1]}{suffix}"] += 1
    return counts


def bond_counts_helper(mol: Any) -> Counter[str]:
    """Return bond type counts with atom hybridization for a molecule that already has explicit hydrogens."""
    counts: Counter[str] = Counter()
    if mol is None:
        return counts
    for bond in mol.GetBonds():
        a1_label = get_atom_state(bond.GetBeginAtom())
        a2_label = get_atom_state(bond.GetEndAtom())
        pair = sorted([a1_label, a2_label])

        btype = bond.GetBondType()
        if btype == Chem.BondType.SINGLE:
            suffix = " (single)"
        elif btype == Chem.BondType.DOUBLE:
            suffix = " (double)"
        elif btype == Chem.BondType.TRIPLE:
            suffix = " (triple)"
        elif btype == Chem.BondType.AROMATIC:
            suffix = " (aromatic)"
        else:
            suffix = f" ({str(btype).lower()})"
        counts[f"{pair[0]}-{pair[1]}{suffix}"] += 1
    return counts


def classify_reaction_type(
    target_smiles: str,
    left_balance_terms: tuple[BalanceTerm, ...],
    rhs_terms: tuple[ReferenceTerm, ...],
    right_balance_terms: tuple[BalanceTerm, ...],
    unresolved_left_atoms: Counter[str],
    unresolved_right_atoms: Counter[str],
) -> str:
    """Determine the classification of the reaction based on conservation of groups and bonds."""
    if unresolved_left_atoms or unresolved_right_atoms:
        return "Unbalanced"

    left: list[tuple[int, str]] = []
    if target_smiles:
        left.append((1, target_smiles))
    left += [(term.count, term.smiles) for term in left_balance_terms if term.count > 0]
    right = [(term.count, term.smiles) for term in rhs_terms if term.count > 0]
    right += [
        (term.count, term.smiles) for term in right_balance_terms if term.count > 0
    ]
    return classify_sides(left, right)


def classify_sides(
    left: Sequence[tuple[int, str]], right: Sequence[tuple[int, str]]
) -> str:
    """Classify a reaction given both sides as (coefficient, SMILES) pairs."""
    _require_rdkit()

    def gather(side: Sequence[tuple[int, str]]) -> list[tuple[Any, int]]:
        mols = []
        for count, smiles in side:
            if count <= 0 or not smiles:
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                mols.append((Chem.AddHs(mol), count))
        return mols

    lhs_mols = gather(left)
    rhs_mols = gather(right)

    # 1. Elemental Balance check
    lhs_atoms: Counter[str] = Counter()
    for mol, count in lhs_mols:
        for atom in mol.GetAtoms():
            lhs_atoms[atom.GetSymbol()] += count

    rhs_atoms: Counter[str] = Counter()
    for mol, count in rhs_mols:
        for atom in mol.GetAtoms():
            rhs_atoms[atom.GetSymbol()] += count

    if lhs_atoms != rhs_atoms:
        return "Unbalanced"

    # 2. Hyperhomodesmotic check (LHS hybridized bonds == RHS hybridized bonds)
    def get_hybridized_bonds(m: Any) -> Counter[str]:
        bonds: Counter[str] = Counter()
        if m is None:
            return bonds
        for bond in m.GetBonds():
            a1 = get_atom_state(bond.GetBeginAtom())
            a2 = get_atom_state(bond.GetEndAtom())
            pair = "-".join(sorted([a1, a2]))
            btype = bond.GetBondType()
            bonds[f"{pair} ({str(btype).lower()})"] += 1
        return bonds

    lhs_hyb_bonds: Counter[str] = Counter()
    for mol, count in lhs_mols:
        for btype, b_count in get_hybridized_bonds(mol).items():
            lhs_hyb_bonds[btype] += b_count * count

    rhs_hyb_bonds: Counter[str] = Counter()
    for mol, count in rhs_mols:
        for btype, b_count in get_hybridized_bonds(mol).items():
            rhs_hyb_bonds[btype] += b_count * count

    if lhs_hyb_bonds == rhs_hyb_bonds:
        return "Hyperhomodesmotic"

    # 3. Homodesmotic check (LHS groups == RHS groups)
    lhs_groups: Counter[str] = Counter()
    for mol, count in lhs_mols:
        for group, grp_count in count_groups(Chem.RemoveHs(mol)).items():
            lhs_groups[group] += grp_count * count

    rhs_groups: Counter[str] = Counter()
    for mol, count in rhs_mols:
        for group, grp_count in count_groups(Chem.RemoveHs(mol)).items():
            rhs_groups[group] += grp_count * count

    if lhs_groups == rhs_groups:
        return "Homodesmotic"

    # 4. Isodesmic check (LHS simple bonds == RHS simple bonds)
    lhs_simple_bonds: Counter[str] = Counter()
    for mol, count in lhs_mols:
        for btype, b_count in simple_bond_counts_helper(mol).items():
            lhs_simple_bonds[btype] += b_count * count

    rhs_simple_bonds: Counter[str] = Counter()
    for mol, count in rhs_mols:
        for btype, b_count in simple_bond_counts_helper(mol).items():
            rhs_simple_bonds[btype] += b_count * count

    if lhs_simple_bonds == rhs_simple_bonds:
        return "Isodesmic"

    return "Elemental Balance"


def explicit_hydrogen_copy(mol: Any) -> Any:
    """Return a copy with explicit hydrogens without mutating the host molecule."""
    _require_rdkit()
    return Chem.AddHs(mol) if mol is not None else None


def unique_substructure_matches(
    mol: Any, smarts: str, atom_centric: bool = False
) -> set[tuple[int, ...]]:
    """Return canonicalized atom-index tuples for one SMARTS pattern."""
    _require_rdkit()
    pattern = Chem.MolFromSmarts(smarts)
    if pattern is None or mol is None:
        return set()
    matches = mol.GetSubstructMatches(pattern, uniquify=False)
    if atom_centric:
        return {(match[0],) for match in matches}
    return {tuple(sorted(match)) for match in matches}


def analyze_molecule(
    mol: Any,
    reference_overrides: Mapping[str, str] | None = None,
    user_species: Sequence[UserSpecies] | None = None,
    excluded_smiles: Sequence[str] | None = None,
) -> AnalysisResult:
    """Analyze a molecule and create a draft homodesmotic equation.

    *reference_overrides* maps an environment name to a replacement reference
    SMILES, *user_species* extends the balance library, and *excluded_smiles*
    takes species out of it -- one whose energy the user cannot compute is no
    use to them however well it balances. All three are stored on the result
    verbatim so the report quotes what was typed. The reaction type is
    re-derived from the resulting equation either way, so a substituted
    reference that breaks the balance is reported as broken rather than
    accepted.
    """
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
            Counter(),
            Counter(),
            "Unbalanced",
            Counter(),
            Counter(),
        )

    overrides = {
        name: smiles.strip()
        for name, smiles in (reference_overrides or {}).items()
        if smiles and smiles.strip() and is_valid_smiles(smiles)
    }
    excluded = tuple(
        dict.fromkeys(
            smiles.strip()
            for smiles in (excluded_smiles or ())
            if smiles and smiles.strip()
        )
    )
    kept_species, _rejected = normalize_user_species(user_species)
    extra_species = tuple(entry.as_balance_species() for entry in kept_species)
    required_smiles = tuple(entry.smiles for entry in kept_species if entry.required)
    required_sides = {
        entry.smiles: entry.side
        for entry in kept_species
        if entry.required and entry.side
    }

    target_smiles = Chem.MolToSmiles(Chem.RemoveHs(mol), isomericSmiles=True)
    target_atoms = atom_counts(explicit_hydrogen_copy(mol))
    target_groups = count_groups(mol)
    reference_atoms: Counter[str] = Counter()
    reference_groups: Counter[str] = Counter()
    matches: list[EnvironmentMatch] = []
    rhs_terms: list[ReferenceTerm] = []

    for rule in ENVIRONMENTS:
        count = len(unique_substructure_matches(mol, rule.smarts, rule.atom_centric))
        if count == 0:
            continue

        ref_smiles = overrides.get(rule.name, rule.reference_smiles)
        overridden = ref_smiles != rule.reference_smiles
        core_indices = reference_original_atom_indices(rule, ref_smiles)
        description = (
            f"User-specified reference (default: {rule.reference_smiles})."
            if overridden
            else rule.description
        )

        matches.append(
            EnvironmentMatch(
                rule.name,
                count,
                ref_smiles,
                description,
                core_indices,
                user_defined=overridden,
            )
        )
        rhs_terms.append(
            ReferenceTerm(
                count,
                ref_smiles,
                core_indices,
                user_defined=overridden,
            )
        )

        ref_mol = Chem.MolFromSmiles(ref_smiles)
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

    unmet_required: tuple[str, ...] = ()
    ignored_exclusions: tuple[str, ...] = ()

    if milp is not None and np is not None:
        reference_side = tuple(term.smiles for term in rhs_terms)
        # An exclusion that leaves nothing balanceable is dropped rather than
        # failing the run, the same way an impossible requirement is.
        for banned in (excluded, ()) if excluded else ((),):
            left_balance_terms, right_balance_terms, milp_success, unmet_required = (
                build_hyperhomodesmotic_balance_terms(
                    group_delta,
                    atom_delta,
                    extra_species,
                    required_smiles,
                    reference_side,
                    banned,
                    required_sides,
                )
            )
            if milp_success:
                ignored_exclusions = () if banned else excluded
                break

    if not milp_success:
        left_needed = Counter(
            {element: count for element, count in atom_delta.items() if count > 0}
        )
        right_needed = Counter(
            {element: -count for element, count in atom_delta.items() if count < 0}
        )
        left_balance_terms, unresolved_left_atoms, unmet_left = build_balance_terms(
            left_needed, extra_species, required_smiles, excluded
        )
        right_balance_terms, unresolved_right_atoms, unmet_right = build_balance_terms(
            right_needed, extra_species, required_smiles, excluded
        )
        # A required species only has to land on one side, so it counts as
        # unmet only when neither side could take it.
        unmet_required = tuple(
            smiles for smiles in unmet_left if smiles in set(unmet_right)
        )

    # Algebraic cancellation of identical reference molecules on both sides
    lhs_counts = Counter()
    for term in left_balance_terms:
        if term.count > 0:
            lhs_counts[term.smiles] += term.count

    rhs_counts = Counter()
    for term in rhs_terms:
        if term.count > 0:
            rhs_counts[term.smiles] += term.count
    for term in right_balance_terms:
        if term.count > 0:
            rhs_counts[term.smiles] += term.count

    cancellations = Counter()
    for smiles, l_count in lhs_counts.items():
        r_count = rhs_counts.get(smiles, 0)
        if r_count > 0:
            cancellations[smiles] = min(l_count, r_count)

    new_left_balance_terms = []
    for term in left_balance_terms:
        cancel_qty = cancellations.get(term.smiles, 0)
        new_count = max(0, term.count - cancel_qty)
        new_left_balance_terms.append(
            BalanceTerm(
                name=term.name,
                smiles=term.smiles,
                count=new_count,
                description=term.description,
                user_defined=term.user_defined,
            )
        )

    new_right_balance_terms = []
    rhs_cancel_remaining = Counter(cancellations)
    for term in right_balance_terms:
        cancel_qty = rhs_cancel_remaining.get(term.smiles, 0)
        if cancel_qty > 0:
            sub = min(term.count, cancel_qty)
            new_count = term.count - sub
            rhs_cancel_remaining[term.smiles] -= sub
        else:
            new_count = term.count
        new_right_balance_terms.append(
            BalanceTerm(
                name=term.name,
                smiles=term.smiles,
                count=new_count,
                description=term.description,
                user_defined=term.user_defined,
            )
        )

    new_rhs_terms = []
    new_matches = []
    for rhs_term, match in zip(rhs_terms, matches):
        cancel_qty = rhs_cancel_remaining.get(rhs_term.smiles, 0)
        if cancel_qty > 0:
            sub = min(rhs_term.count, cancel_qty)
            new_count = rhs_term.count - sub
            rhs_cancel_remaining[rhs_term.smiles] -= sub
        else:
            new_count = rhs_term.count
        new_rhs_terms.append(
            ReferenceTerm(
                count=new_count,
                smiles=rhs_term.smiles,
                original_atom_indices=rhs_term.original_atom_indices,
                user_defined=rhs_term.user_defined,
            )
        )
        new_matches.append(
            EnvironmentMatch(
                name=match.name,
                count=new_count,
                reference_smiles=match.reference_smiles,
                description=match.description,
                original_atom_indices=match.original_atom_indices,
                user_defined=match.user_defined,
            )
        )

    # A reference the solver had to put on both sides cancels to nothing. The
    # remaining equation can still be perfectly balanced while no longer saying
    # anything about the environment it was built to measure, so name it.
    cancelled_references = tuple(
        before.name
        for before, after in zip(matches, new_matches)
        if before.count > 0 and after.count == 0
    )

    left_balance_terms = tuple(new_left_balance_terms)
    right_balance_terms = tuple(new_right_balance_terms)
    rhs_terms = tuple(new_rhs_terms)
    matches = tuple(new_matches)

    # Recalculate reference_atoms and atom_delta based on cancelled terms
    reference_atoms = Counter()
    for term in rhs_terms:
        if term.count > 0:
            ref_mol = Chem.MolFromSmiles(term.smiles)
            if ref_mol is not None:
                for element, number in atom_counts(
                    explicit_hydrogen_copy(ref_mol)
                ).items():
                    reference_atoms[element] += number * term.count

    atom_delta = Counter()
    for element in sorted(set(target_atoms) | set(reference_atoms)):
        delta = reference_atoms[element] - target_atoms[element]
        if delta != 0:
            atom_delta[element] = delta

    target_bonds = bond_counts(mol)

    lhs_bonds = Counter(target_bonds)
    for term in left_balance_terms:
        if term.count > 0:
            b_mol = Chem.MolFromSmiles(term.smiles)
            if b_mol is not None:
                for btype, bcount in bond_counts(b_mol).items():
                    lhs_bonds[btype] += bcount * term.count

    rhs_bonds = Counter()
    for term in rhs_terms:
        if term.count > 0:
            b_mol = Chem.MolFromSmiles(term.smiles)
            if b_mol is not None:
                for btype, bcount in bond_counts(b_mol).items():
                    rhs_bonds[btype] += bcount * term.count

    for term in right_balance_terms:
        if term.count > 0:
            b_mol = Chem.MolFromSmiles(term.smiles)
            if b_mol is not None:
                for btype, bcount in bond_counts(b_mol).items():
                    rhs_bonds[btype] += bcount * term.count

    reaction_bonds_delta = Counter()
    for btype in sorted(set(lhs_bonds) | set(rhs_bonds)):
        diff = rhs_bonds[btype] - lhs_bonds[btype]
        if diff != 0:
            reaction_bonds_delta[btype] = diff

    reaction_type = classify_reaction_type(
        target_smiles,
        left_balance_terms,
        rhs_terms,
        right_balance_terms,
        unresolved_left_atoms,
        unresolved_right_atoms,
    )

    user_input_lines = describe_user_input(
        tuple(sorted(overrides.items())),
        kept_species,
        unmet_required,
        cancelled_references,
        excluded,
        ignored_exclusions,
    )

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
        target_bonds,
        reaction_bonds_delta,
        reaction_type,
        lhs_bonds,
        rhs_bonds,
        user_input_lines,
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
        target_bonds,
        reaction_bonds_delta,
        reaction_type,
        lhs_bonds,
        rhs_bonds,
        user_input_lines,
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
        target_bonds,
        reaction_bonds_delta,
        reaction_type,
        lhs_bonds,
        rhs_bonds,
        tuple(sorted(overrides.items())),
        kept_species,
        unmet_required,
        cancelled_references,
        excluded,
        ignored_exclusions,
    )


def species_atom_counts(smiles: str) -> Counter[str]:
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return Counter()
    return atom_counts(explicit_hydrogen_copy(mol))


def is_valid_smiles(smiles: str) -> bool:
    """Return True when RDKit can parse *smiles*."""
    if Chem is None or not smiles or not smiles.strip():
        return False
    return Chem.MolFromSmiles(smiles.strip()) is not None


def normalize_user_species(
    entries: Sequence[UserSpecies] | None,
) -> tuple[tuple[UserSpecies, ...], tuple[str, ...]]:
    """Split user entries into parseable ones and the SMILES that failed.

    Duplicated SMILES collapse onto the first entry, keeping ``required`` if any
    copy asked for it, so the solver never gets two identical columns.
    """
    kept: list[UserSpecies] = []
    rejected: list[str] = []
    seen: dict[str, int] = {}
    for entry in entries or ():
        smiles = entry.smiles.strip()
        if not smiles:
            continue
        if not is_valid_smiles(smiles):
            rejected.append(entry.smiles)
            continue
        if smiles in seen:
            index = seen[smiles]
            if entry.required and not kept[index].required:
                kept[index] = UserSpecies(
                    kept[index].smiles, kept[index].name, True, entry.side
                )
            continue
        seen[smiles] = len(kept)
        kept.append(UserSpecies(smiles, entry.name.strip(), entry.required, entry.side))
    return tuple(kept), tuple(rejected)


def reference_original_atom_indices(
    rule: EnvironmentRule, reference_smiles: str | None = None
) -> tuple[int, ...]:
    """Return reference atoms that correspond to the original matched environment."""
    _require_rdkit()
    ref_mol = Chem.MolFromSmiles(
        rule.reference_smiles if reference_smiles is None else reference_smiles
    )
    pattern = Chem.MolFromSmarts(rule.smarts)
    if ref_mol is None or pattern is None:
        return ()

    matches = ref_mol.GetSubstructMatches(pattern)
    if not matches:
        return ()
    if rule.atom_centric:
        return (matches[0][0],)
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
    return tuple(
        sorted((element, count) for element, count in values.items() if count > 0)
    )


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
    min_heavy = (
        min(
            sum(
                count
                for element, count in candidates[index][1].items()
                if element != "H"
            )
            for index in indexes
        )
        if indexes
        else 0
    )
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
        best_score: tuple | None = None
        for index in range(start_index, len(candidates)):
            _species, counts = candidates[index]
            if not _counter_fits(counts, remaining):
                continue

            next_remaining = _subtract_counter(remaining, counts)
            tail = search(_counter_key(next_remaining), index)
            if tail is None:
                continue

            solution = (index, *tail)
            score = _solution_score(solution, candidates)
            if best_score is None or score < best_score:
                best = solution
                best_score = score

        return best

    return search(_counter_key(needed_atoms), 0)


def balance_pool(
    extra_species: Sequence[BalanceSpecies] = (),
    excluded_smiles: Sequence[str] = (),
) -> tuple[tuple[BalanceSpecies, ...], frozenset[str]]:
    """Built-in species plus the user's, and the SMILES the user contributed.

    A user entry whose SMILES already exists in the library reuses that entry
    instead of adding a duplicate column, but is still reported as theirs.
    Anything in *excluded_smiles* is left out of the pool entirely: a species
    whose energy the user cannot compute is no use in their reaction.
    """
    banned = {smiles for smiles in excluded_smiles if smiles}
    pool = [species for species in BALANCE_SPECIES if species.smiles not in banned]
    known = {species.smiles for species in pool}
    user_smiles = set()
    for species in extra_species:
        if species.smiles in banned:
            continue
        user_smiles.add(species.smiles)
        if species.smiles not in known:
            known.add(species.smiles)
            pool.append(species)
    return tuple(pool), frozenset(user_smiles)


def build_hyperhomodesmotic_balance_terms(
    group_delta: Counter[str],
    atom_delta: Counter[str] | None = None,
    extra_species: Sequence[BalanceSpecies] = (),
    required_smiles: Sequence[str] = (),
    reference_smiles: Sequence[str] = (),
    excluded_smiles: Sequence[str] = (),
    required_sides: Mapping[str, str] | None = None,
) -> tuple[tuple[BalanceTerm, ...], tuple[BalanceTerm, ...], bool, tuple[str, ...]]:
    """Solve for a perfectly balanced hyperhomodesmotic reaction using MILP.

    *atom_delta* is the per-element shortfall the balance species must make
    up (reference atoms minus target atoms). Constraining it explicitly is
    what stops the solver introducing an element that appears nowhere in the
    reaction: group descriptors alone cannot express "conserve silicon", so
    without this the solver was free to balance cyclopropane by adding
    tetramethylsilane to one side.
    """
    pool, user_smiles = balance_pool(extra_species, excluded_smiles)
    # Keep the user's order: it decides which requirement yields to which when
    # they cannot all be met.
    required = tuple(dict.fromkeys(smiles for smiles in required_smiles if smiles))

    if not group_delta and not required:
        return (), (), True, ()

    species_list = []
    for species in pool:
        mol = Chem.MolFromSmiles(species.smiles)
        if mol is not None:
            species_list.append(
                (species, count_groups(mol), species_atom_counts(species.smiles))
            )

    if not species_list:
        return (), (), False, tuple(required)

    group_names = sorted(
        set(k for _, c, _ in species_list for k in c.keys()) | set(group_delta.keys())
    )
    atom_delta = Counter(atom_delta or {})
    element_names = sorted(
        set(e for _, _, a in species_list for e in a.keys()) | set(atom_delta.keys())
    )
    n_groups = len(group_names)
    n_species = len(species_list)

    A = np.zeros((n_groups + len(element_names), n_species))
    for j, (_, c, a) in enumerate(species_list):
        for i, name in enumerate(group_names):
            A[i, j] = c[name]
        for i, element in enumerate(element_names):
            A[n_groups + i, j] = a[element]

    A_eq = np.hstack((A, -A))
    b_eq = np.array(
        [group_delta.get(name, 0) for name in group_names]
        + [atom_delta.get(element, 0) for element in element_names]
    )

    c = np.ones(2 * n_species)
    for j, (species, _groups, _atoms) in enumerate(species_list):
        heavy = sum(1 for char in species.smiles if char.isalpha() and char != "H")
        cost = 1.0
        if heavy == 1:
            cost = 1.02
        elif heavy == 2:
            cost = 1.01
        c[j] = cost
        c[n_species + j] = cost

    integrality = np.ones(2 * n_species)
    constraints = [LinearConstraint(A_eq, b_eq, b_eq)]

    row_of = {species.smiles: j for j, (species, _g, _a) in enumerate(species_list)}
    required_rows = [row_of[smiles] for smiles in required if smiles in row_of]
    unplaceable = tuple(smiles for smiles in required if smiles not in row_of)
    sides = {
        row_of[smiles]: side
        for smiles, side in (required_sides or {}).items()
        if smiles in row_of
    }

    # A reference molecule added to the left just cancels the copies on the
    # right, and the cancellation pass then erases both. That is often the
    # cheapest solution and always a useless one: it leaves a balanced equation
    # with nothing left to say about the strained environment. Keep references
    # off the left, and only relax it if nothing else balances at all.
    protected = [row_of[smiles] for smiles in set(reference_smiles) if smiles in row_of]

    # A requirement that cannot be met is dropped rather than failing the whole
    # run: a good balance the user cannot have still beats no balance at all.
    # Dropped one at a time, newest first, so a species that is fine on its own
    # is not blamed for a later one that is not.
    res = None
    kept: list[int] = []
    dropped: list[int] = []
    for banned in (protected, ()) if protected else ((),):
        kept = list(required_rows)
        dropped = []
        res = _solve_with_required(
            c, constraints, integrality, n_species, kept, banned, sides
        )
        while res is None and kept:
            dropped.append(kept.pop())
            res = _solve_with_required(
                c, constraints, integrality, n_species, kept, banned, sides
            )
        if res is None:
            continue

        # Dropping newest-first can discard a requirement that was fine on its
        # own, so offer each dropped one back once. One extra solve each, and
        # the answer no longer depends on the order they were added in.
        for row in list(reversed(dropped)):
            trial = _solve_with_required(
                c, constraints, integrality, n_species, kept + [row], banned, sides
            )
            if trial is not None:
                kept.append(row)
                dropped.remove(row)
                res = trial
        break

    if res is None:
        return (), (), False, unplaceable

    unmet: tuple[str, ...] = unplaceable + tuple(
        species_list[j][0].smiles for j in required_rows if j in set(dropped)
    )

    # The tail past 2*n_species holds the side-picking binaries, not counts.
    u = np.round(res.x[:n_species]).astype(int)
    v = np.round(res.x[n_species : 2 * n_species]).astype(int)

    left_terms = []
    for j, count in enumerate(u):
        if count > 0:
            species = species_list[j][0]
            left_terms.append(
                BalanceTerm(
                    species.name,
                    species.smiles,
                    int(count),
                    description=species.description,
                    user_defined=species.smiles in user_smiles,
                )
            )

    right_terms = []
    for j, count in enumerate(v):
        if count > 0:
            species = species_list[j][0]
            right_terms.append(
                BalanceTerm(
                    species.name,
                    species.smiles,
                    int(count),
                    description=species.description,
                    user_defined=species.smiles in user_smiles,
                )
            )

    return tuple(left_terms), tuple(right_terms), True, unmet


#: Upper bound on any single species count, used to switch a side on and off.
#: Far above anything a balance needs, and small enough to keep the solver
#: numerically comfortable.
_SIDE_BIG_M = 10_000


def _solve_with_required(
    c: Any,
    constraints: list,
    integrality: Any,
    n_species: int,
    required_rows: Sequence[int],
    banned_left_rows: Sequence[int] = (),
    required_sides: Mapping[int, str] | None = None,
) -> Any:
    """Cheapest MILP solution, pinning each required species to one side.

    Asking only for "present somewhere" lets the solver put a species on both
    sides, where it cancels out and the user sees no trace of it. One binary
    per required species picks its side instead:

        u - M*(1-b) <= 0     b = 0 forces the right side empty
        v - M*b     <= 0     b = 1 forces the left side empty
        u + v       >= 1     and something has to be there

    Choosing sides this way is one solve rather than one per assignment, so a
    handful of required species costs no more than a single one. Naming a side
    in *required_sides* pins that binary instead of leaving it to the solver.
    """
    required_sides = required_sides or {}
    n_required = len(required_rows)
    n_vars = 2 * n_species + n_required

    def widen(constraint: Any) -> Any:
        if n_required == 0:
            return constraint
        matrix = np.hstack(
            (constraint.A, np.zeros((constraint.A.shape[0], n_required)))
        )
        return LinearConstraint(matrix, constraint.lb, constraint.ub)

    all_constraints = [widen(constraint) for constraint in constraints]

    if n_required:
        rows = np.zeros((3 * n_required, n_vars))
        lb = np.empty(3 * n_required)
        ub = np.empty(3 * n_required)
        for t, j in enumerate(required_rows):
            binary = 2 * n_species + t

            rows[3 * t, j] = 1
            rows[3 * t, binary] = _SIDE_BIG_M
            lb[3 * t], ub[3 * t] = -np.inf, _SIDE_BIG_M

            rows[3 * t + 1, n_species + j] = 1
            rows[3 * t + 1, binary] = -_SIDE_BIG_M
            lb[3 * t + 1], ub[3 * t + 1] = -np.inf, 0

            rows[3 * t + 2, j] = 1
            rows[3 * t + 2, n_species + j] = 1
            lb[3 * t + 2], ub[3 * t + 2] = 1, np.inf
        all_constraints.append(LinearConstraint(rows, lb, ub))

    costs = np.concatenate((c, np.zeros(n_required)))
    upper = np.concatenate((np.full(2 * n_species, np.inf), np.ones(n_required)))
    lower = np.zeros(n_vars)
    for row in banned_left_rows:
        upper[row] = 0
    for t, j in enumerate(required_rows):
        side = required_sides.get(j)
        if side == SIDE_LEFT:
            upper[2 * n_species + t] = 0
        elif side == SIDE_RIGHT:
            lower[2 * n_species + t] = 1
    bounds = Bounds(lower, upper)

    try:
        res = milp(
            c=costs,
            constraints=all_constraints,
            integrality=np.ones(n_vars),
            bounds=bounds,
        )
    except Exception:
        return None
    return res if res.success else None


def build_balance_terms(
    needed_atoms: Counter[str],
    extra_species: Sequence[BalanceSpecies] = (),
    required_smiles: Sequence[str] = (),
    excluded_smiles: Sequence[str] = (),
) -> tuple[tuple[BalanceTerm, ...], Counter[str], tuple[str, ...]]:
    """Build simple molecule terms that exactly cover the requested atom counts.

    Returns the terms, any atoms left over, and the required SMILES that could
    not be placed on this side.
    """
    pool, user_smiles = balance_pool(extra_species, excluded_smiles)

    # Seed one copy of each required species before searching. This side only
    # has to absorb a specific shortfall, so a species the shortfall cannot
    # afford belongs on the other side, not here.
    remaining = Counter(needed_atoms)
    seeded: Counter[BalanceSpecies] = Counter()
    unmet: list[str] = []
    by_smiles = {species.smiles: species for species in pool}
    for smiles in required_smiles:
        species = by_smiles.get(smiles)
        if species is None:
            unmet.append(smiles)
            continue
        counts = species_atom_counts(smiles)
        if counts and _counter_fits(counts, remaining):
            remaining = _subtract_counter(remaining, counts)
            seeded[species] += 1
        else:
            unmet.append(smiles)

    if not remaining:
        return _balance_terms_from(seeded, user_smiles), Counter(), tuple(unmet)

    allowed_elements = set(remaining)
    candidates: list[tuple[BalanceSpecies, Counter[str]]] = []
    for species in pool:
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
    solution = _search_balance(remaining, candidate_tuple)
    if solution is None:
        return (
            _balance_terms_from(seeded, user_smiles),
            Counter(remaining),
            tuple(unmet),
        )

    grouped = Counter(seeded)
    grouped.update(candidate_tuple[index][0] for index in solution)
    return _balance_terms_from(grouped, user_smiles), Counter(), tuple(unmet)


def _balance_terms_from(
    grouped: Counter[BalanceSpecies], user_smiles: frozenset[str]
) -> tuple[BalanceTerm, ...]:
    return tuple(
        BalanceTerm(
            species.name,
            species.smiles,
            count,
            description=species.description,
            user_defined=species.smiles in user_smiles,
        )
        for species, count in sorted(grouped.items(), key=lambda item: item[0].name)
        if count > 0
    )


def describe_user_input(
    reference_overrides: tuple[tuple[str, str], ...],
    user_species: Sequence[UserSpecies],
    unmet_required: Sequence[str] = (),
    cancelled_references: Sequence[str] = (),
    excluded_species: Sequence[str] = (),
    ignored_exclusions: Sequence[str] = (),
) -> tuple[str, ...]:
    """One line per user entry, quoting the SMILES exactly as it was typed."""
    lines: list[str] = []
    for name, smiles in reference_overrides:
        lines.append(f"Reference override - {name}: {smiles}")
    for entry in user_species:
        label = f" ({entry.name})" if entry.name else ""
        if entry.required and entry.side:
            suffix = f" [required, {entry.side}]"
        elif entry.required:
            suffix = " [required]"
        else:
            suffix = ""
        lines.append(f"User species - {entry.smiles}{label}{suffix}")
    for smiles in unmet_required:
        lines.append(f"Required species could not be placed: {smiles}")
    for smiles in excluded_species:
        if smiles in ignored_exclusions:
            continue
        lines.append(f"Excluded species - {smiles}")
    for name in cancelled_references:
        lines.append(
            f"Reference for '{name}' cancelled out; the equation no longer "
            "covers that environment"
        )
    for smiles in ignored_exclusions:
        lines.append(f"Nothing balances without {smiles}, so it was used anyway")
    return tuple(lines)


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
    target_bonds: Counter[str],
    reaction_bonds_delta: Counter[str],
    reaction_type: str,
    lhs_bonds: Counter[str] | None = None,
    rhs_bonds: Counter[str] | None = None,
    user_input_lines: Sequence[str] = (),
) -> str:
    if lhs_bonds is None:
        lhs_bonds = Counter()
    if rhs_bonds is None:
        rhs_bonds = Counter()
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
    is_hyper = reaction_type == "Hyperhomodesmotic"
    hyper_status = "Satisfied" if is_hyper else "Not satisfied"
    bond_status_lines = []
    for btype in sorted(set(lhs_bonds) | set(rhs_bonds)):
        l_c = lhs_bonds[btype]
        r_c = rhs_bonds[btype]
        status = (
            f"Satisfied (LHS: {l_c}, RHS: {r_c})"
            if l_c == r_c
            else f"Not satisfied (LHS: {l_c}, RHS: {r_c}, Delta: {r_c - l_c:+})"
        )
        bond_status_lines.append(f"  - {btype}: {status}")
    bond_status_text = (
        "\n".join(bond_status_lines) if bond_status_lines else "  - No bonds detected"
    )

    user_input_text = (
        "\n".join(f"  - {line}" for line in user_input_lines)
        if user_input_lines
        else "  - none (all species are the built-in defaults)"
    )

    return (
        "Homodesmotic Draft Equation\n"
        "===========================\n\n"
        f"{lhs} -> {rhs}\n\n"
        f"User-specified input:\n{user_input_text}\n\n"
        f"Reaction type: {reaction_type}\n"
        f"Hyperhomodesmotic condition: {hyper_status}\n"
        f"Bond-type conservation status:\n{bond_status_text}\n\n"
        f"Target atom count: {format_counter(target_atoms)}\n"
        f"Reference-side atom count: {format_counter(reference_atoms)}\n"
        f"Reference minus target atom delta: {format_counter(atom_delta)}\n\n"
        f"Target bond counts: {format_counter(target_bonds)}\n"
        f"Left-side bond counts: {format_counter(lhs_bonds)}\n"
        f"Right-side bond counts: {format_counter(rhs_bonds)}\n"
        f"Reaction bond difference: {format_counter(reaction_bonds_delta)}\n\n"
        f"Left-side balancing species: {left_balance}\n"
        f"Right-side balancing species: {right_balance}\n"
        f"Unresolved left-side atoms: {unresolved_left}\n"
        f"Unresolved right-side atoms: {unresolved_right}\n\n"
        "Energy expression template\n"
        "--------------------------\n"
        "Strain or stabilization energy can be estimated from a balanced reaction:\n"
        "SE = sum(E_products) - sum(E_reactants)\n\n"
        "Notes\n"
        "-----\n"
        f"{SELF_CHECK_NOTE}\n\n"
        "This tool detects predefined local environments and proposes reference "
        "molecules. Simple balancing species are added automatically when the "
        "atom-count difference can be represented by the built-in molecule library.\n\n"
        "Common balancing species and reference molecules appearing on both sides "
        "of the equation have been algebraically cancelled to prevent redundant "
        "reference calculations in downstream quantum chemistry pipelines."
    )


def format_terms(terms: Iterable[BalanceTerm]) -> str:
    parts = [
        f"{term.count} {term.smiles} ({term.name})"
        + (" [user]" if term.user_defined else "")
        for term in terms
        if term.count > 0
    ]
    return " + ".join(parts) if parts else "none"


def format_reference_terms(terms: Iterable[ReferenceTerm]) -> str:
    parts = [
        f"{term.count} {term.smiles}" + (" [user]" if term.user_defined else "")
        for term in terms
        if term.count > 0
    ]
    return " + ".join(parts) if parts else "no reference molecules detected"


#: Everything the user typed is drawn in one colour so it stands out from the
#: automatically chosen species.
USER_COLOR = "#ff8a65"


def _span(text: str, color: str, title: str = "") -> str:
    safe_text = html.escape(text)
    safe_title = html.escape(title)
    title_attr = f' title="{safe_title}"' if safe_title else ""
    return (
        f'<span style="color:{color}; font-weight:600;"{title_attr}>{safe_text}</span>'
    )


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
            color, title = indexed_colors.get(
                atom_index, (default_color, default_title)
            )
            pieces.append(_span(two_letter, color, title))
            atom_index += 1
            index += 2
            continue

        if char in organic_one_letter_atoms:
            color, title = indexed_colors.get(
                atom_index, (default_color, default_title)
            )
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
    if term.user_defined:
        smiles = color_smiles_atoms(
            term.smiles, USER_COLOR, "User-specified reference molecule"
        )
        return f"{term.count} {smiles}"

    """Color atom tokens of a reference fragment.

    True core atoms are colored with original_color (blue), while any capping
    atoms are colored with added_color (green).
    """
    cores = _BALANCE_CORE_MAP.get(term.smiles, None)
    if cores is not None:
        # Use the full balance-core-map definition as the true core atoms.
        # original_atom_indices only captures the SMARTS-matched atoms, which for
        # atom-centric aromatic rules is just 1 atom — not the whole ring environment.
        # _BALANCE_CORE_MAP is the authoritative source of which atoms are core.
        true_core_atoms = list(cores)
    else:
        # No map entry: fall back to all SMARTS-matched atoms as environment atoms.
        true_core_atoms = sorted(term.original_atom_indices)

    indexed_colors = {}
    for atom_idx in true_core_atoms:
        indexed_colors[atom_idx] = (
            original_color,
            "Original strain-environment atom in reference fragment",
        )

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


def _html_terms(
    terms: Iterable[BalanceTerm],
    core_color: str,
    added_color: str,
) -> str:
    parts = []
    for term in terms:
        if term.count > 0:
            if term.user_defined:
                parts.append(
                    f"{term.count} "
                    + color_smiles_atoms(
                        term.smiles, USER_COLOR, "User-specified species"
                    )
                    + f' <span style="color:#9aa0a6;">({html.escape(term.name)}, '
                    "user)</span>"
                )
                continue
            core_indices = _BALANCE_CORE_MAP.get(term.smiles, ())
            indexed_colors = {
                idx: (core_color, "Balance species core environment atom")
                for idx in core_indices
            }
            smiles = color_smiles_atoms_by_index(
                term.smiles,
                indexed_colors,
                added_color,
                "Added balancing atom",
            )
            parts.append(
                f"{term.count} {smiles} "
                f'<span style="color:#9aa0a6;">({html.escape(term.name)})</span>'
            )
    return " + ".join(parts) if parts else "none"


def get_reaction_type_color(rtype: str) -> str:
    if rtype == "Hyperhomodesmotic":
        return "#81c995"  # Vibrant light green
    elif rtype == "Homodesmotic":
        return "#8ab4f8"  # Sky blue
    elif rtype == "Isodesmic":
        return "#fde293"  # Yellow
    elif rtype == "Elemental Balance":
        return "#c58af9"  # Purple
    return "#f28b82"  # Red / pink for Unbalanced


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
    target_bonds: Counter[str] | None = None,
    reaction_bonds_delta: Counter[str] | None = None,
    reaction_type: str = "Unbalanced",
    lhs_bonds: Counter[str] | None = None,
    rhs_bonds: Counter[str] | None = None,
    user_input_lines: Sequence[str] = (),
) -> str:
    """Build a styled HTML representation of the homodesmotic equation.

    Uses specific color-coding logic:
      - Blue: Target core atoms and reference core atoms
      - Yellow: Core atoms of the balancing species (adjusting over-counted bonds)
      - Green: Capping atoms on LHS and LHS balance species
      - Purple: RHS balance species
      - Red: Unresolved species/atoms
    """
    target_color = "#8ab4f8"
    reference_color = "#8ab4f8"
    balance_core_color = "#fde293"
    left_added_color = "#81c995"
    right_added_color = "#c58af9"
    unresolved_color = "#f28b82"

    if target_bonds is None:
        target_bonds = Counter()
    if reaction_bonds_delta is None:
        reaction_bonds_delta = Counter()
    if lhs_bonds is None:
        lhs_bonds = Counter()
    if rhs_bonds is None:
        rhs_bonds = Counter()

    left_balance = _html_terms(left_balance_terms, balance_core_color, left_added_color)
    right_balance = _html_terms(
        right_balance_terms, balance_core_color, right_added_color
    )
    unresolved_left = _html_counter(
        unresolved_left_atoms, unresolved_color, "Unresolved left-side atom"
    )
    unresolved_right = _html_counter(
        unresolved_right_atoms, unresolved_color, "Unresolved right-side atom"
    )

    target_label = target_smiles or "Target"
    lhs_parts = [color_smiles_atoms(target_label, target_color, "Original target atom")]
    if left_balance != "none":
        lhs_parts.append(left_balance)
    if unresolved_left != "none":
        lhs_parts.append(
            f"{_span('unresolved left species', unresolved_color)} ({unresolved_left})"
        )

    rhs_parts = [
        " + ".join(
            color_reference_term(
                term,
                reference_color,
                left_added_color,
            )
            for term in rhs_terms
            if term.count > 0
        )
        or "no reference molecules detected"
    ]
    if right_balance != "none":
        rhs_parts.append(right_balance)
    if unresolved_right != "none":
        rhs_parts.append(
            f"{_span('unresolved right species', unresolved_color)} ({unresolved_right})"
        )

    if user_input_lines:
        user_input_html = (
            '<ul style="margin:4px 0 0 16px; padding:0; list-style-type:disc;'
            f' color:{USER_COLOR};">'
            + "".join(f"<li>{html.escape(line)}</li>" for line in user_input_lines)
            + "</ul>"
        )
    else:
        user_input_html = (
            '<p style="margin:4px 0 0 16px; color:#9aa0a6;">'
            "none (all species are the built-in defaults)</p>"
        )

    warning_html = ""
    if is_elemental_balance:
        warning_html = '<p style="color:#fde293; font-weight:bold;">⚠️ Note: Calculated in elemental balance mode due to environment matching constraints.</p>'

    is_hyper = reaction_type == "Hyperhomodesmotic"
    hyper_status_html = (
        '<span style="color:#81c995; font-weight:bold;">Satisfied</span>'
        if is_hyper
        else '<span style="color:#f28b82; font-weight:bold;">Not satisfied</span>'
    )
    bond_status_htmls = []
    for btype in sorted(set(lhs_bonds) | set(rhs_bonds)):
        l_c = lhs_bonds[btype]
        r_c = rhs_bonds[btype]
        if l_c == r_c:
            status_html = f'<span style="color:#81c995; font-weight:bold;">Satisfied</span> (LHS: {l_c}, RHS: {r_c})'
        else:
            status_html = f'<span style="color:#f28b82; font-weight:bold;">Not satisfied</span> (LHS: {l_c}, RHS: {r_c}, Delta: {r_c - l_c:+})'
        bond_status_htmls.append(
            f'<li><span style="color:#e8eaed;">{btype}</span>: {status_html}</li>'
        )

    if bond_status_htmls:
        bond_status_list_html = (
            '<ul style="margin: 4px 0 0 16px; padding: 0; list-style-type: disc; color: #9aa0a6;">'
            + "".join(bond_status_htmls)
            + "</ul>"
        )
    else:
        bond_status_list_html = (
            '<p style="margin: 4px 0 0 16px; color: #9aa0a6;">No bonds detected.</p>'
        )

    return (
        '<div style="font-family:Consolas, monospace; color:#e8eaed;">'
        '<h3 style="margin:0 0 8px 0; color:#e8eaed;">Homodesmotic Draft Equation</h3>'
        f"{warning_html}"
        f'<p style="font-size:16px; font-weight:500; line-height:1.6; margin:16px 0;">{" + ".join(lhs_parts)} '
        f'<span style="color:#e8eaed;">-&gt;</span> {" + ".join(rhs_parts)}</p>'
        '<hr style="border:0; border-top:1px solid #3c4043;">'
        f'<div style="margin-bottom:12px;"><span style="color:#9aa0a6;">'
        f"User-specified input:</span>{user_input_html}</div>"
        f'<p><span style="color:#9aa0a6;">Reaction Type:</span> '
        f'<span style="color:{get_reaction_type_color(reaction_type)}; font-weight:bold;">{reaction_type}</span></p>'
        f'<p><span style="color:#9aa0a6;">Hyperhomodesmotic condition:</span> {hyper_status_html}</p>'
        f'<div style="margin-bottom:12px;"><span style="color:#9aa0a6;">Bond-type conservation status:</span>{bond_status_list_html}</div>'
        f'<p><span style="color:#9aa0a6;">Target atom count:</span> '
        f"{_html_counter(target_atoms, '#e8eaed', 'Target atom')}</p>"
        f'<p><span style="color:#9aa0a6;">Reference-side atom count:</span> '
        f"{_html_counter(reference_atoms, '#e8eaed', 'Reference atom')}</p>"
        f'<p><span style="color:#9aa0a6;">Reference minus target atom delta:</span> '
        f"{_html_counter(atom_delta, '#e8eaed', 'Atom-count difference')}</p>"
        f'<p><span style="color:#9aa0a6;">Target bond counts:</span> '
        f"{_html_counter(target_bonds, '#e8eaed', 'Target bond')}</p>"
        f'<p><span style="color:#9aa0a6;">Left-side bond counts:</span> '
        f"{_html_counter(lhs_bonds, '#e8eaed', 'Left-side bond')}</p>"
        f'<p><span style="color:#9aa0a6;">Right-side bond counts:</span> '
        f"{_html_counter(rhs_bonds, '#e8eaed', 'Right-side bond')}</p>"
        f'<p><span style="color:#9aa0a6;">Reaction bond difference:</span> '
        f"{_html_counter(reaction_bonds_delta, '#e8eaed', 'Reaction bond difference')}</p>"
        f'<p><span style="color:#9aa0a6;">Left-side balancing species:</span> {left_balance}</p>'
        f'<p><span style="color:#9aa0a6;">Right-side balancing species:</span> {right_balance}</p>'
        f'<p><span style="color:#9aa0a6;">Unresolved left-side atoms:</span> {unresolved_left}</p>'
        f'<p><span style="color:#9aa0a6;">Unresolved right-side atoms:</span> {unresolved_right}</p>'
        '<p style="color:#9aa0a6;">Blue = original target and reference cores, '
        "yellow = balancing cores to adjust over-counted bonds, "
        "green = automatically added left balance species and caps, "
        "purple = automatically added right balance species, "
        "orange = user-specified, "
        "red = unresolved.</p>"
        '<p style="color:#9aa0a6; font-size:12px; margin-top:8px;">'
        "Note: Common balancing species and reference molecules appearing on both sides "
        "of the equation have been algebraically cancelled to prevent redundant reference calculations.</p>"
        f'<p style="color:#fde293; font-size:12px; margin-top:8px;">'
        f"{html.escape(SELF_CHECK_NOTE)}</p>"
        "</div>"
    )


@dataclass(frozen=True)
class EquationCheck:
    """What a hand-written equation conserves, and what it does not."""

    left: tuple[tuple[int, str], ...]
    right: tuple[tuple[int, str], ...]
    reaction_type: str
    atom_delta: Counter[str]
    group_delta: Counter[str]
    bond_delta: Counter[str]
    lhs_bonds: Counter[str]
    rhs_bonds: Counter[str]
    errors: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.errors


#: Everything people type or paste for the arrow, longest first: replacing
#: "->" before "-->" would leave the extra dash behind on the left-hand side.
_ARROWS = tuple(
    sorted(
        ("->", "=>", "-->", "==>", "→", "⟶", "➔"),
        key=len,
        reverse=True,
    )
)


def _split_on_plus(text: str) -> list[str]:
    """Split on ``+`` at bracket depth zero.

    A charge lives inside brackets -- ``[NH4+]`` -- so a naive split would cut
    a perfectly good SMILES in half.
    """
    parts: list[str] = []
    depth = 0
    current: list[str] = []
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth = max(0, depth - 1)
        if char == "+" and depth == 0:
            parts.append("".join(current))
            current = []
            continue
        current.append(char)
    parts.append("".join(current))
    return parts


def _quiet_smiles(smiles: str) -> Any:
    """Parse without RDKit narrating every rejected candidate to stderr."""
    if Chem is None or not smiles:
        return None
    try:
        from rdkit import rdBase

        with rdBase.BlockLogs():
            return Chem.MolFromSmiles(smiles)
    except (ImportError, AttributeError):  # pragma: no cover - old RDKit
        return Chem.MolFromSmiles(smiles)


def _read_species(term: str) -> tuple[Any, str]:
    """Read a term as a SMILES, ignoring any name after it.

    A SMILES never contains whitespace, so the first token is the molecule and
    the rest is a label -- which is how the report writes it, ``CCC (propane)``,
    and how RDKit itself reads a SMILES file. Trusting RDKit to reject the tail
    would not work: it happily parses ``CCC (propane)`` as propane *named*
    "(propane)", and the annotation would then be kept as part of the SMILES.
    """
    candidate = term.split()[0] if term.split() else ""
    return _quiet_smiles(candidate), candidate


def parse_equation_side(text: str) -> tuple[list[tuple[int, str]], list[str]]:
    """Parse one side into (coefficient, SMILES) pairs, plus any complaints."""
    terms: list[tuple[int, str]] = []
    errors: list[str] = []
    for raw in _split_on_plus(text):
        term = raw.strip()
        if not term:
            continue
        match = re.match(r"^(\d+)\s+(.*)$", term)
        if match:
            count, rest = int(match.group(1)), match.group(2)
        else:
            count, rest = 1, term
        if count == 0:
            continue
        mol, smiles = _read_species(rest)
        if mol is None:
            errors.append(f"Cannot read '{term}' as a molecule.")
            continue
        stray = _stray_molecule(rest)
        if stray:
            errors.append(f"Missing '+' before '{stray}'?")
            continue
        terms.append((count, smiles))
    return terms, errors


def _stray_molecule(term: str) -> str:
    """A second molecule sitting in one term, i.e. a forgotten ``+``.

    Everything after the SMILES is treated as its name, so a missing ``+``
    would otherwise drop a whole species and hand back a confident verdict on
    the wrong reaction. A real name is not mistaken for one: "(propane)" and
    "ethane" do not parse as SMILES.
    """
    tail = term.split()[1:]
    for index, token in enumerate(tail):
        if token.startswith(("(", "#", "//")):
            return ""
        if _quiet_smiles(token) is not None:
            return " ".join(tail[index:])
    return ""


def parse_equation(
    text: str,
) -> tuple[list[tuple[int, str]], list[tuple[int, str]], list[str]]:
    """Parse ``A + 2 B -> 3 C`` into both sides.

    Accepts a line copied straight out of the report, coefficients and all.
    """
    cleaned = (text or "").strip()
    for arrow in _ARROWS:
        cleaned = cleaned.replace(arrow, "\n->\n")
    pieces = [piece.strip() for piece in cleaned.split("\n->\n")]
    if len(pieces) == 1:
        return [], [], ["No arrow found. Write the reaction as 'A + B -> C'."]
    if len(pieces) > 2:
        return [], [], ["More than one arrow found."]

    left, left_errors = parse_equation_side(pieces[0])
    right, right_errors = parse_equation_side(pieces[1])
    errors = left_errors + right_errors
    if not left:
        errors.append("Nothing on the left of the arrow.")
    if not right:
        errors.append("Nothing on the right of the arrow.")
    return left, right, errors


def _side_totals(side: Sequence[tuple[int, str]], counter: Any) -> Counter[str]:
    totals: Counter[str] = Counter()
    for count, smiles in side:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        for key, value in counter(mol).items():
            totals[key] += count * value
    return totals


def _delta(left: Counter[str], right: Counter[str]) -> Counter[str]:
    delta: Counter[str] = Counter()
    for key in sorted(set(left) | set(right)):
        difference = right[key] - left[key]
        if difference:
            delta[key] = difference
    return delta


def check_equation(text: str) -> EquationCheck:
    """Judge an equation the user typed, by the same rules as a generated one."""
    _require_rdkit()
    left, right, errors = parse_equation(text)
    if errors:
        return EquationCheck(
            tuple(left),
            tuple(right),
            "Unbalanced",
            Counter(),
            Counter(),
            Counter(),
            Counter(),
            Counter(),
            tuple(errors),
        )

    atoms_left = _side_totals(left, lambda m: atom_counts(explicit_hydrogen_copy(m)))
    atoms_right = _side_totals(right, lambda m: atom_counts(explicit_hydrogen_copy(m)))
    groups_left = _side_totals(left, count_groups)
    groups_right = _side_totals(right, count_groups)
    bonds_left = _side_totals(left, bond_counts)
    bonds_right = _side_totals(right, bond_counts)

    return EquationCheck(
        tuple(left),
        tuple(right),
        classify_sides(left, right),
        _delta(atoms_left, atoms_right),
        _delta(groups_left, groups_right),
        _delta(bonds_left, bonds_right),
        bonds_left,
        bonds_right,
        (),
    )


def format_equation_sides(
    left: Sequence[tuple[int, str]], right: Sequence[tuple[int, str]]
) -> str:
    """Render both sides as the one-line form the tester reads back."""

    def side(terms: Sequence[tuple[int, str]]) -> str:
        return " + ".join(
            smiles if count == 1 else f"{count} {smiles}" for count, smiles in terms
        )

    return f"{side(left)} -> {side(right)}"


def build_equation_check_text(check: EquationCheck) -> str:
    """Plain-text verdict on a hand-written equation."""
    if check.errors:
        return "Equation Check\n==============\n\n" + "\n".join(
            f"  - {error}" for error in check.errors
        )

    def verdict(name: str, delta: Counter[str]) -> str:
        if not delta:
            return f"{name}: conserved"
        detail = ", ".join(f"{key}: {value:+}" for key, value in sorted(delta.items()))
        return f"{name}: NOT conserved ({detail})"

    return (
        "Equation Check\n"
        "==============\n\n"
        f"{format_equation_sides(check.left, check.right)}\n\n"
        f"Reaction type: {check.reaction_type}\n"
        f"{verdict('Atoms', check.atom_delta)}\n"
        f"{verdict('Groups', check.group_delta)}\n"
        f"{verdict('Bond types', check.bond_delta)}\n\n"
        f"Left-side bond counts: {format_counter(check.lhs_bonds)}\n"
        f"Right-side bond counts: {format_counter(check.rhs_bonds)}\n\n"
        "Deltas are right minus left. Atoms must balance for any reaction; "
        "groups as well for a homodesmotic one; bond types as well for a "
        f"hyperhomodesmotic one.\n\n{SELF_CHECK_NOTE}"
    )


def build_equation_check_html(check: EquationCheck) -> str:
    """Styled verdict, matching the draft report."""
    if check.errors:
        items = "".join(f"<li>{html.escape(error)}</li>" for error in check.errors)
        return (
            '<div style="font-family:Consolas, monospace; color:#e8eaed;">'
            '<h3 style="margin:0 0 8px 0;">Equation Check</h3>'
            f'<ul style="color:#f28b82;">{items}</ul>'
            "</div>"
        )

    def row(name: str, delta: Counter[str]) -> str:
        if not delta:
            state = '<span style="color:#81c995; font-weight:bold;">conserved</span>'
            detail = ""
        else:
            state = (
                '<span style="color:#f28b82; font-weight:bold;">not conserved</span>'
            )
            detail = (
                " ("
                + ", ".join(
                    f"{html.escape(key)}: {value:+}"
                    for key, value in sorted(delta.items())
                )
                + ")"
            )
        return f'<li><span style="color:#9aa0a6;">{name}</span>: {state}{detail}</li>'

    equation = " + ".join(
        color_smiles_atoms(
            smiles if count == 1 else f"{count} {smiles}",
            "#8ab4f8",
            "Left-side species",
        )
        for count, smiles in check.left
    )
    equation += ' <span style="color:#e8eaed;">-&gt;</span> '
    equation += " + ".join(
        color_smiles_atoms(
            smiles if count == 1 else f"{count} {smiles}",
            "#c58af9",
            "Right-side species",
        )
        for count, smiles in check.right
    )

    return (
        '<div style="font-family:Consolas, monospace; color:#e8eaed;">'
        '<h3 style="margin:0 0 8px 0; color:#e8eaed;">Equation Check</h3>'
        f'<p style="font-size:16px; line-height:1.6; margin:16px 0;">{equation}</p>'
        '<hr style="border:0; border-top:1px solid #3c4043;">'
        f'<p><span style="color:#9aa0a6;">Reaction Type:</span> '
        f'<span style="color:{get_reaction_type_color(check.reaction_type)}; '
        f'font-weight:bold;">{html.escape(check.reaction_type)}</span></p>'
        '<ul style="margin:4px 0 12px 16px; padding:0; list-style-type:disc;">'
        + row("Atoms", check.atom_delta)
        + row("Groups", check.group_delta)
        + row("Bond types", check.bond_delta)
        + "</ul>"
        f'<p><span style="color:#9aa0a6;">Left-side bond counts:</span> '
        f"{_html_counter(check.lhs_bonds, '#e8eaed', 'Left-side bond')}</p>"
        f'<p><span style="color:#9aa0a6;">Right-side bond counts:</span> '
        f"{_html_counter(check.rhs_bonds, '#e8eaed', 'Right-side bond')}</p>"
        '<p style="color:#9aa0a6; font-size:12px;">Deltas are right minus left. '
        "Atoms must balance for any reaction; groups as well for a homodesmotic "
        "one; bond types as well for a hyperhomodesmotic one.</p>"
        f'<p style="color:#fde293; font-size:12px;">{html.escape(SELF_CHECK_NOTE)}</p>'
        "</div>"
    )


def _user_input_card(result: AnalysisResult) -> str:
    """HTML card quoting the user's overrides and species as they were typed."""
    lines = describe_user_input(
        result.reference_overrides,
        result.user_species,
        result.unmet_required,
        result.cancelled_references,
        result.excluded_species,
        result.ignored_exclusions,
    )
    if not lines:
        return ""
    items = "".join(f"<li>{html.escape(line)}</li>" for line in lines)
    return (
        '        <div class="card" style="padding-top: 20px;">\n'
        '            <h3 style="margin: 0 0 12px 0; font-size: 18px; '
        'font-weight: 500;">User-specified input</h3>\n'
        # The report body is grey and the equation above already colours
        # these entries, so shouting them again just makes the card noisy.
        '            <ul style="margin:0 0 0 18px; padding:0; '
        'color:var(--text-muted);">'
        f"{items}</ul>\n"
        "        </div>\n"
    )


def export_analysis(
    path: str | Path,
    result: AnalysisResult,
    current_file_path: str | Path | None = None,
) -> None:
    """Export analysis output as HTML, CSV, or plain text."""
    output_path = Path(path)
    filename = Path(current_file_path).name if current_file_path else "Untitled"
    generated_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if output_path.suffix.lower() in {".html", ".htm"}:
        from . import PLUGIN_VERSION

        output_path.write_text(
            "<!doctype html>\n"
            "<html>\n"
            "<head>\n"
            '    <meta charset="utf-8">\n'
            f"    <title>Strain Homodesmotic Reaction Report (v{PLUGIN_VERSION})</title>\n"
            '    <link rel="preconnect" href="https://fonts.googleapis.com">\n'
            '    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>\n'
            '    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">\n'
            "    <style>\n"
            "        :root {\n"
            "            --bg-color: #121214;\n"
            "            --card-bg: #1e1e24;\n"
            "            --text-main: #e8eaed;\n"
            "            --text-muted: #9aa0a6;\n"
            "            --border-color: #2f3037;\n"
            "            --blue-color: #8ab4f8;\n"
            "            --green-color: #81c995;\n"
            "            --purple-color: #c58af9;\n"
            "            --blue-bg: rgba(138, 180, 248, 0.12);\n"
            "            --green-bg: rgba(129, 201, 149, 0.12);\n"
            "            --purple-bg: rgba(197, 138, 249, 0.12);\n"
            "        }\n"
            "        body {\n"
            "            background-color: var(--bg-color);\n"
            "            color: var(--text-main);\n"
            "            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;\n"
            "            margin: 0;\n"
            "            padding: 40px 24px;\n"
            "            display: flex;\n"
            "            justify-content: center;\n"
            "        }\n"
            "        .container {\n"
            "            max-width: 960px;\n"
            "            width: 100%;\n"
            "        }\n"
            "        .report-header {\n"
            "            margin-bottom: 24px;\n"
            "            border-bottom: 1px solid var(--border-color);\n"
            "            padding-bottom: 16px;\n"
            "        }\n"
            "        h1 {\n"
            "            font-size: 24px;\n"
            "            font-weight: 600;\n"
            "            margin: 0 0 8px 0;\n"
            "            color: var(--blue-color);\n"
            "        }\n"
            "        .meta-info {\n"
            "            font-size: 13px;\n"
            "            color: var(--text-muted);\n"
            "            margin-bottom: 4px;\n"
            "        }\n"
            "        .card {\n"
            "            background-color: var(--card-bg);\n"
            "            border: 1px solid var(--border-color);\n"
            "            border-radius: 8px;\n"
            "            padding: 20px;\n"
            "            margin-bottom: 20px;\n"
            "        }\n"
            "        table {\n"
            "            width: 100%;\n"
            "            border-collapse: separate;\n"
            "            border-spacing: 0;\n"
            "            margin-top: 16px;\n"
            "            border: 1px solid var(--border-color);\n"
            "            border-radius: 8px;\n"
            "            overflow: hidden;\n"
            "        }\n"
            "        th, td {\n"
            "            padding: 12px 16px;\n"
            "            text-align: left;\n"
            "            font-size: 14px;\n"
            "        }\n"
            "        th {\n"
            "            background-color: #25262c;\n"
            "            color: var(--text-main);\n"
            "            font-weight: 600;\n"
            "            border-bottom: 1px solid var(--border-color);\n"
            "        }\n"
            "        td {\n"
            "            border-bottom: 1px solid rgba(47, 48, 55, 0.5);\n"
            "        }\n"
            "        tr:last-child td {\n"
            "            border-bottom: none;\n"
            "        }\n"
            "        .row-ref {\n"
            "            background-color: var(--blue-bg);\n"
            "            color: var(--blue-color);\n"
            "        }\n"
            "        .row-left {\n"
            "            background-color: var(--green-bg);\n"
            "            color: var(--green-color);\n"
            "        }\n"
            "        .row-right {\n"
            "            background-color: var(--purple-bg);\n"
            "            color: var(--purple-color);\n"
            "        }\n"
            "        .smiles-mono {\n"
            "            font-family: 'JetBrains Mono', monospace;\n"
            "            font-size: 13px;\n"
            "        }\n"
            "    </style>\n"
            "</head>\n"
            "<body>\n"
            '    <div class="container">\n'
            '        <div class="report-header">\n'
            "            <h1>Homodesmotic Reaction Report</h1>\n"
            f'            <div class="meta-info">File: {html.escape(filename)}</div>\n'
            f'            <div class="meta-info">Generated on: {generated_date}</div>\n'
            f'            <div class="meta-info">Generated by Strain Homodesmotic Reaction Generator plugin v{PLUGIN_VERSION}</div>\n'
            "        </div>\n"
            "        \n"
            '        <div class="card">\n'
            f"            {result.equation_html}\n"
            "        </div>\n"
            "        \n"
            '        <div class="card" style="padding-top: 20px;">\n'
            '            <h3 style="margin: 0 0 12px 0; font-size: 18px; font-weight: 500;">Reaction Components</h3>\n'
            "            <table>\n"
            "                <thead>\n"
            "                    <tr>\n"
            "                        <th>Type</th>\n"
            "                        <th>Environment / Species</th>\n"
            "                        <th>Count</th>\n"
            "                        <th>SMILES</th>\n"
            "                        <th>Description</th>\n"
            "                    </tr>\n"
            "                </thead>\n"
            "                <tbody>\n"
            + "\n".join(
                '                    <tr class="row-ref">'
                "<td>Ref</td>"
                f"<td>{html.escape(match.name)}</td>"
                f"<td>{match.count}</td>"
                f'<td class="smiles-mono">{html.escape(match.reference_smiles)}</td>'
                f"<td>{html.escape(match.description)}</td>"
                "</tr>"
                for match in result.matches
                if match.count > 0
            )
            + "\n".join(
                '                    <tr class="row-left">'
                "<td>Left Balance</td>"
                f"<td>{html.escape(term.name)}</td>"
                f"<td>{term.count}</td>"
                f'<td class="smiles-mono">{html.escape(term.smiles)}</td>'
                f"<td>{html.escape(term.description or 'Added left-side balance species')}</td>"
                "</tr>"
                for term in result.left_balance_terms
                if term.count > 0
            )
            + "\n"
            + "\n".join(
                '                    <tr class="row-right">'
                "<td>Right Balance</td>"
                f"<td>{html.escape(term.name)}</td>"
                f"<td>{term.count}</td>"
                f'<td class="smiles-mono">{html.escape(term.smiles)}</td>'
                f"<td>{html.escape(term.description or 'Added right-side balance species')}</td>"
                "</tr>"
                for term in result.right_balance_terms
                if term.count > 0
            )
            + "\n                </tbody>\n"
            "            </table>\n"
            "        </div>\n" + _user_input_card(result) + "    </div>\n"
            "</body>\n"
            "</html>\n",
            encoding="utf-8",
        )
        return

    if output_path.suffix.lower() == ".txt":
        header = (
            f"Homodesmotic Reaction Report\n"
            f"============================\n"
            f"File: {filename}\n"
            f"Generated on: {generated_date}\n\n"
        )
        output_path.write_text(header + result.equation_text + "\n", encoding="utf-8")
        return

    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Homodesmotic Reaction Report"])
        writer.writerow(["File", filename])
        writer.writerow(["Generated on", generated_date])
        writer.writerow([])
        writer.writerow(["Environment", "Count", "Reference SMILES", "Description"])
        for match in result.matches:
            if match.count > 0:
                writer.writerow(
                    [match.name, match.count, match.reference_smiles, match.description]
                )
        writer.writerow(["Reaction Type", result.reaction_type])
        writer.writerow(
            [
                "Hyperhomodesmotic condition",
                "Satisfied"
                if result.reaction_type == "Hyperhomodesmotic"
                else "Not satisfied",
            ]
        )
        writer.writerow([])
        writer.writerow(["Bond-type conservation status"])
        writer.writerow(["Bond Type", "LHS Count", "RHS Count", "Delta", "Status"])
        for btype in sorted(set(result.lhs_bonds) | set(result.rhs_bonds)):
            l_c = result.lhs_bonds[btype]
            r_c = result.rhs_bonds[btype]
            delta = r_c - l_c
            status = "Satisfied" if l_c == r_c else "Not satisfied"
            writer.writerow(
                [btype, l_c, r_c, f"{delta:+}" if delta != 0 else "0", status]
            )
        writer.writerow([])
        writer.writerow(["Target atom count", format_counter(result.target_atoms)])
        writer.writerow(
            ["Reference-side atom count", format_counter(result.reference_atoms)]
        )
        writer.writerow(
            ["Reference minus target atom delta", format_counter(result.atom_delta)]
        )
        writer.writerow(["Target bond counts", format_counter(result.target_bonds)])
        writer.writerow(["Left-side bond counts", format_counter(result.lhs_bonds)])
        writer.writerow(["Right-side bond counts", format_counter(result.rhs_bonds)])
        writer.writerow(
            ["Reaction bond difference", format_counter(result.reaction_bonds_delta)]
        )
        writer.writerow(
            ["Left-side balancing species", format_terms(result.left_balance_terms)]
        )
        writer.writerow(
            ["Right-side balancing species", format_terms(result.right_balance_terms)]
        )
        writer.writerow(
            ["Unresolved left-side atoms", format_counter(result.unresolved_left_atoms)]
        )
        writer.writerow(
            [
                "Unresolved right-side atoms",
                format_counter(result.unresolved_right_atoms),
            ]
        )
        writer.writerow([])
        writer.writerow(["User-specified input"])
        lines = describe_user_input(
            result.reference_overrides,
            result.user_species,
            result.unmet_required,
            result.cancelled_references,
            result.excluded_species,
            result.ignored_exclusions,
        )
        if lines:
            for line in lines:
                writer.writerow([line])
        else:
            writer.writerow(["none (all species are the built-in defaults)"])
        writer.writerow([])
        writer.writerow(["Equation"])
        writer.writerow([result.equation_text])
