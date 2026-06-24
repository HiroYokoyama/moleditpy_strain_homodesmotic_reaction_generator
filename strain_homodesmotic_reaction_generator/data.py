#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Data structures and constant databases for Strain Homodesmotic Reaction Generator.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EnvironmentRule:
    name: str
    smarts: str
    reference_smiles: str
    description: str
    atom_centric: bool = False


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
    EnvironmentRule(
        "Primary-Carbonyl",
        "[CX4H3]-[CX3](=[OX1])",
        "CC(=O)C",
        "Carbonyl carbon bonded to primary carbons.",
    ),
    EnvironmentRule(
        "Secondary-Carbonyl",
        "[CX4H2]-[CX3](=[OX1])",
        "CCC(=O)C",
        "Carbonyl carbon bonded to a secondary carbon.",
    ),
    EnvironmentRule(
        "Tertiary-Carbonyl",
        "[CX4H1]-[CX3](=[OX1])",
        "CC(C)C(=O)C",
        "Carbonyl carbon bonded to a tertiary carbon.",
    ),
    EnvironmentRule(
        "Quaternary-Carbonyl",
        "[CX4H0]-[CX3](=[OX1])",
        "CC(C)(C)C(=O)C",
        "Carbonyl carbon bonded to a quaternary carbon.",
    ),
    # Amines
    EnvironmentRule(
        "Primary-Primary-Amine",
        "[CX4H3]-[NX3H2]",
        "CN",
        "Primary amine bonded to primary carbon.",
    ),
    EnvironmentRule(
        "Secondary-Primary-Amine",
        "[CX4H2]-[NX3H2]",
        "CCN",
        "Primary amine bonded to secondary carbon.",
    ),
    EnvironmentRule(
        "Tertiary-Primary-Amine",
        "[CX4H1]-[NX3H2]",
        "CC(C)N",
        "Primary amine bonded to tertiary carbon.",
    ),
    EnvironmentRule(
        "Quaternary-Primary-Amine",
        "[CX4H0]-[NX3H2]",
        "CC(C)(C)N",
        "Primary amine bonded to quaternary carbon.",
    ),
    EnvironmentRule(
        "Primary-Secondary-Amine",
        "[CX4H3]-[NX3H1]",
        "CNC",
        "Secondary amine bonded to primary carbon.",
    ),
    EnvironmentRule(
        "Secondary-Secondary-Amine",
        "[CX4H2]-[NX3H1]",
        "CCNC",
        "Secondary amine bonded to secondary carbon.",
    ),
    EnvironmentRule(
        "Tertiary-Secondary-Amine",
        "[CX4H1]-[NX3H1]",
        "CC(C)NC",
        "Secondary amine bonded to tertiary carbon.",
    ),
    EnvironmentRule(
        "Quaternary-Secondary-Amine",
        "[CX4H0]-[NX3H1]",
        "CC(C)(C)NC",
        "Secondary amine bonded to quaternary carbon.",
    ),
    EnvironmentRule(
        "Primary-Tertiary-Amine",
        "[CX4H3]-[NX3H0]",
        "CN(C)C",
        "Tertiary amine bonded to primary carbon.",
    ),
    EnvironmentRule(
        "Secondary-Tertiary-Amine",
        "[CX4H2]-[NX3H0]",
        "CCN(C)C",
        "Tertiary amine bonded to secondary carbon.",
    ),
    EnvironmentRule(
        "Tertiary-Tertiary-Amine",
        "[CX4H1]-[NX3H0]",
        "CC(C)N(C)C",
        "Tertiary amine bonded to tertiary carbon.",
    ),
    EnvironmentRule(
        "Quaternary-Tertiary-Amine",
        "[CX4H0]-[NX3H0]",
        "CC(C)(C)N(C)C",
        "Tertiary amine bonded to quaternary carbon.",
    ),
    # Alkenes
    EnvironmentRule(
        "Secondary-alkene", "[CX3H2]=[CX3H2]", "C=C", "Ethene-like double bond."
    ),
    EnvironmentRule(
        "Tertiary-alkene", "[CX3H1]=[CX3H2]", "CC=C", "Propene-like double bond."
    ),
    EnvironmentRule(
        "Quaternary-alkene", "[CX3H0]=[CX3H2]", "CC(=C)C", "Isobutene-like double bond."
    ),
    EnvironmentRule(
        "Internal-alkene", "[CX3H1]=[CX3H1]", "CC=CC", "2-Butene-like double bond."
    ),
    EnvironmentRule(
        "Internal-branched-alkene",
        "[CX3H0]=[CX3H1]",
        "CC=C(C)C",
        "2-Methyl-2-butene-like double bond.",
    ),
    EnvironmentRule(
        "Tetrasubstituted-alkene",
        "[CX3H0]=[CX3H0]",
        "CC(C)=C(C)C",
        "Tetramethylethene-like double bond.",
    ),
    # Alkynes & Nitriles
    EnvironmentRule(
        "Terminal-alkyne", "[CX2H1]#[CX2H1]", "C#C", "Ethyne-like triple bond."
    ),
    EnvironmentRule(
        "Substituted-alkyne", "[CX2H1]#[CX2H0]", "CC#C", "Propyne-like triple bond."
    ),
    EnvironmentRule(
        "Internal-alkyne", "[CX2H0]#[CX2H0]", "CC#CC", "2-Butyne-like triple bond."
    ),
    EnvironmentRule(
        "Nitrile", "[CX2H0]#[NX1]", "CC#N", "Acetonitrile-like triple bond."
    ),
    # Aromatics
    EnvironmentRule(
        "Aromatic-CH",
        "[cX3H1]",
        "c1ccccc1",
        "Aromatic carbon-hydrogen bond in benzene ring.",
    ),
    EnvironmentRule(
        "Aromatic-C-Aromatic",
        "[cX3H0]-[c]",
        "c1ccc(-c2ccccc2)cc1",
        "Aromatic carbon bonded to another aromatic carbon (biaryl linkage).",
        atom_centric=True,
    ),
    EnvironmentRule(
        "Aromatic-C-Carbon",
        "[cX3H0]-[!a&#6]",
        "Cc1ccccc1",
        "Aromatic carbon bonded to a non-aromatic carbon.",
        atom_centric=True,
    ),
    EnvironmentRule(
        "Aromatic-fusion-carbon",
        "[cX3H0](:[c])(:[c]):[c]",
        "c1ccc2ccccc2c1",
        "Aromatic fusion carbon in naphthalene ring.",
        atom_centric=True,
    ),
)


BALANCE_SPECIES: tuple[BalanceSpecies, ...] = (
    BalanceSpecies("methane", "C", "One-carbon saturated hydrocarbon balance species."),
    BalanceSpecies("ethane", "CC", "Two-carbon saturated hydrocarbon balance species."),
    BalanceSpecies(
        "propane", "CCC", "Three-carbon saturated hydrocarbon balance species."
    ),
    BalanceSpecies(
        "butane", "CCCC", "Four-carbon saturated hydrocarbon balance species."
    ),
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
    BalanceSpecies("Ethene", "C=C", "Simplest alkene"),
    BalanceSpecies("Propene", "CC=C", "Three-carbon alkene"),
    BalanceSpecies("Isobutene", "CC(=C)C", "Branched alkene"),
    BalanceSpecies("2-Butene", "CC=CC", "Internal alkene"),
    BalanceSpecies("2-Methyl-2-butene", "CC=C(C)C", "Trisubstituted alkene"),
    BalanceSpecies("2,3-Dimethyl-2-butene", "CC(C)=C(C)C", "Tetrasubstituted alkene"),
    BalanceSpecies("Ethyne", "C#C", "Simplest alkyne"),
    BalanceSpecies("Propyne", "CC#C", "Three-carbon alkyne"),
    BalanceSpecies("2-Butyne", "CC#CC", "Internal alkyne"),
    BalanceSpecies("Acetonitrile", "CC#N", "Simplest nitrile"),
    BalanceSpecies("Benzene", "c1ccccc1", "Simplest aromatic hydrocarbon"),
    BalanceSpecies("Toluene", "Cc1ccccc1", "Methyl-substituted benzene"),
    BalanceSpecies(
        "Biphenyl", "c1ccc(-c2ccccc2)cc1", "Simplest biaryl aromatic hydrocarbon"
    ),
    BalanceSpecies(
        "Naphthalene", "c1ccc2ccccc2c1", "Fused two-ring aromatic hydrocarbon"
    ),
)


_BALANCE_CORE_MAP: dict[str, tuple[int, ...]] = {
    "C": (),
    "CC": (),
    "CCC": (1,),
    "CCCC": (1, 2),
    "CCCCC": (1, 2, 3),
    "CCCCCC": (1, 2, 3, 4),
    "COC": (1,),
    "CCOC": (1, 2),
    "CCOCC": (1, 2, 3),
    "CC(C)C": (1,),
    "CC(C)(C)C": (1,),
    "COC(C)C": (1, 2),
    "COC(C)(C)C": (1, 2),
    "C=O": (0, 1),
    "CC=O": (1, 2),
    "CC(=O)C": (1, 2),
    "N": (0,),
    "CN": (1,),
    "CNC": (1,),
    "CN(C)C": (1,),
    "C=C": (0, 1),
    "CC=C": (1, 2),
    "CC(=C)C": (1, 2),
    "CC=CC": (1, 2),
    "CC=C(C)C": (1, 2),
    "CC(C)=C(C)C": (1, 3),
    "C#C": (0, 1),
    "CC#C": (1, 2),
    "CC#CC": (1, 2),
    "CC#N": (1, 2),
    "c1ccccc1": (0, 1, 2, 3, 4, 5),
    "Cc1ccccc1": (1, 2, 3, 4, 5, 6),
    "c1ccc(-c2ccccc2)cc1": (4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 10, 11),
    "c1ccc2ccccc2c1": (0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
}
