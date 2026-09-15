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


@dataclass(frozen=True)
class UserSpecies:
    """A balance species typed by the user.

    ``smiles`` is kept verbatim so reports quote what was entered rather than
    RDKit's canonical form. ``required`` forces the solver to use it.
    """

    smiles: str
    name: str = ""
    required: bool = False

    def as_balance_species(self) -> "BalanceSpecies":
        return BalanceSpecies(
            self.name or self.smiles,
            self.smiles,
            "User-defined species",
        )


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
    # --- Extended element coverage ------------------------------------
    # Sulfur, phosphorus, silicon, boron, selenium, the halogens, alcohols
    # and the aromatic heterocycles. Before these, anything outside C/H/N/O
    # fell through to an elemental-only balance and was reported as
    # "Unbalanced" -- thiophene, THT, fluorocyclopropane and silolane all
    # produced no usable reaction. Every rule below is checked by
    # tests/test_element_coverage.py to actually match its own reference.
    EnvironmentRule(
        "primary carbon - thioether sulfur - primary carbon",
        "[CX4H3]-[SX2H0]-[CX4H3]",
        "CSC",
        "Thioether sulfur bonded to primary and primary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - thioether sulfur - secondary carbon",
        "[CX4H3]-[SX2H0]-[CX4H2]",
        "CSCC",
        "Thioether sulfur bonded to primary and secondary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - thioether sulfur - tertiary carbon",
        "[CX4H3]-[SX2H0]-[CX4H1]",
        "CSC(C)C",
        "Thioether sulfur bonded to primary and tertiary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - thioether sulfur - quaternary carbon",
        "[CX4H3]-[SX2H0]-[CX4H0]",
        "CSC(C)(C)C",
        "Thioether sulfur bonded to primary and quaternary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - thioether sulfur - secondary carbon",
        "[CX4H2]-[SX2H0]-[CX4H2]",
        "CCSCC",
        "Thioether sulfur bonded to secondary and secondary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - thioether sulfur - tertiary carbon",
        "[CX4H2]-[SX2H0]-[CX4H1]",
        "CCSC(C)C",
        "Thioether sulfur bonded to secondary and tertiary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - thioether sulfur - quaternary carbon",
        "[CX4H2]-[SX2H0]-[CX4H0]",
        "CCSC(C)(C)C",
        "Thioether sulfur bonded to secondary and quaternary carbons.",
    ),
    EnvironmentRule(
        "tertiary carbon - thioether sulfur - tertiary carbon",
        "[CX4H1]-[SX2H0]-[CX4H1]",
        "CC(C)SC(C)C",
        "Thioether sulfur bonded to tertiary and tertiary carbons.",
    ),
    EnvironmentRule(
        "tertiary carbon - thioether sulfur - quaternary carbon",
        "[CX4H1]-[SX2H0]-[CX4H0]",
        "CC(C)SC(C)(C)C",
        "Thioether sulfur bonded to tertiary and quaternary carbons.",
    ),
    EnvironmentRule(
        "quaternary carbon - thioether sulfur - quaternary carbon",
        "[CX4H0]-[SX2H0]-[CX4H0]",
        "CC(C)(C)SC(C)(C)C",
        "Thioether sulfur bonded to quaternary and quaternary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - thiol sulfur",
        "[CX4H3]-[SX2H1]",
        "CS",
        "Thiol S-H on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - thiol sulfur",
        "[CX4H2]-[SX2H1]",
        "CCS",
        "Thiol S-H on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - thiol sulfur",
        "[CX4H1]-[SX2H1]",
        "CC(C)S",
        "Thiol S-H on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - thiol sulfur",
        "[CX4H0]-[SX2H1]",
        "CC(C)(C)S",
        "Thiol S-H on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - fluorine",
        "[CX4H3]-[FX1]",
        "CF",
        "C-F bond on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - fluorine",
        "[CX4H2]-[FX1]",
        "CCF",
        "C-F bond on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - fluorine",
        "[CX4H1]-[FX1]",
        "CC(C)F",
        "C-F bond on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - fluorine",
        "[CX4H0]-[FX1]",
        "CC(C)(C)F",
        "C-F bond on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - chlorine",
        "[CX4H3]-[ClX1]",
        "CCl",
        "C-Cl bond on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - chlorine",
        "[CX4H2]-[ClX1]",
        "CCCl",
        "C-Cl bond on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - chlorine",
        "[CX4H1]-[ClX1]",
        "CC(C)Cl",
        "C-Cl bond on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - chlorine",
        "[CX4H0]-[ClX1]",
        "CC(C)(C)Cl",
        "C-Cl bond on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - bromine",
        "[CX4H3]-[BrX1]",
        "CBr",
        "C-Br bond on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - bromine",
        "[CX4H2]-[BrX1]",
        "CCBr",
        "C-Br bond on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - bromine",
        "[CX4H1]-[BrX1]",
        "CC(C)Br",
        "C-Br bond on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - bromine",
        "[CX4H0]-[BrX1]",
        "CC(C)(C)Br",
        "C-Br bond on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - iodine",
        "[CX4H3]-[IX1]",
        "CI",
        "C-I bond on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - iodine",
        "[CX4H2]-[IX1]",
        "CCI",
        "C-I bond on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - iodine",
        "[CX4H1]-[IX1]",
        "CC(C)I",
        "C-I bond on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - iodine",
        "[CX4H0]-[IX1]",
        "CC(C)(C)I",
        "C-I bond on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - phosphine phosphorus",
        "[CX4H3]-[PX3]",
        "CP",
        "C-P bond on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - phosphine phosphorus",
        "[CX4H2]-[PX3]",
        "CCP",
        "C-P bond on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - phosphine phosphorus",
        "[CX4H1]-[PX3]",
        "CC(C)P",
        "C-P bond on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - phosphine phosphorus",
        "[CX4H0]-[PX3]",
        "CC(C)(C)P",
        "C-P bond on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - silicon",
        "[CX4H3]-[SiX4]",
        "C[SiH3]",
        "C-Si bond on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - silicon",
        "[CX4H2]-[SiX4]",
        "CC[SiH3]",
        "C-Si bond on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - silicon",
        "[CX4H1]-[SiX4]",
        "CC(C)[SiH3]",
        "C-Si bond on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - silicon",
        "[CX4H0]-[SiX4]",
        "CC(C)(C)[SiH3]",
        "C-Si bond on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - boron",
        "[CX4H3]-[BX3]",
        "CB(C)C",
        "C-B bond on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - boron",
        "[CX4H2]-[BX3]",
        "CCB(C)C",
        "C-B bond on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - boron",
        "[CX4H1]-[BX3]",
        "CC(C)B(C)C",
        "C-B bond on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - boron",
        "[CX4H0]-[BX3]",
        "CC(C)(C)B(C)C",
        "C-B bond on a quaternary carbon.",
    ),
    EnvironmentRule(
        "primary carbon - selenoether selenium - primary carbon",
        "[CX4H3]-[SeX2H0]-[CX4H3]",
        "C[Se]C",
        "Selenoether selenium bonded to two primary carbons.",
    ),
    EnvironmentRule(
        "sulfoxide sulfur",
        "[#6]-[SX3](=[OX1])-[#6]",
        "CS(=O)C",
        "Sulfoxide S=O flanked by two carbons.",
    ),
    EnvironmentRule(
        "sulfone sulfur",
        "[#6]-[SX4](=[OX1])(=[OX1])-[#6]",
        "CS(=O)(=O)C",
        "Sulfone SO2 flanked by two carbons.",
    ),
    EnvironmentRule(
        "primary carbon - hydroxyl oxygen",
        "[CX4H3]-[OX2H1]",
        "CO",
        "Alcohol C-OH on a primary carbon.",
    ),
    EnvironmentRule(
        "secondary carbon - hydroxyl oxygen",
        "[CX4H2]-[OX2H1]",
        "CCO",
        "Alcohol C-OH on a secondary carbon.",
    ),
    EnvironmentRule(
        "tertiary carbon - hydroxyl oxygen",
        "[CX4H1]-[OX2H1]",
        "CC(C)O",
        "Alcohol C-OH on a tertiary carbon.",
    ),
    EnvironmentRule(
        "quaternary carbon - hydroxyl oxygen",
        "[CX4H0]-[OX2H1]",
        "CC(C)(C)O",
        "Alcohol C-OH on a quaternary carbon.",
    ),
    EnvironmentRule(
        "Aromatic-furan-oxygen",
        "[oX2]",
        "c1ccoc1",
        "Furan-type aromatic ring oxygen.",
    ),
    EnvironmentRule(
        "Aromatic-thiophene-sulfur",
        "[sX2]",
        "c1ccsc1",
        "Thiophene-type aromatic ring sulfur.",
    ),
    EnvironmentRule(
        "Aromatic-pyrrole-nitrogen",
        "[nX3H1]",
        "c1cc[nH]c1",
        "Pyrrole-type aromatic N-H.",
    ),
    EnvironmentRule(
        "Aromatic-pyridine-nitrogen",
        "[nX2]",
        "c1ccncc1",
        "Pyridine-type aromatic ring nitrogen.",
    ),
    EnvironmentRule(
        "primary carbon - silane silicon - primary carbon",
        "[CX4H3]-[SiX4]-[CX4H3]",
        "C[SiH2]C",
        "Silane silicon bonded to primary and primary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - silane silicon - secondary carbon",
        "[CX4H3]-[SiX4]-[CX4H2]",
        "C[SiH2]CC",
        "Silane silicon bonded to primary and secondary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - silane silicon - tertiary carbon",
        "[CX4H3]-[SiX4]-[CX4H1]",
        "C[SiH2]C(C)C",
        "Silane silicon bonded to primary and tertiary carbons.",
    ),
    EnvironmentRule(
        "primary carbon - silane silicon - quaternary carbon",
        "[CX4H3]-[SiX4]-[CX4H0]",
        "C[SiH2]C(C)(C)C",
        "Silane silicon bonded to primary and quaternary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - silane silicon - secondary carbon",
        "[CX4H2]-[SiX4]-[CX4H2]",
        "CC[SiH2]CC",
        "Silane silicon bonded to secondary and secondary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - silane silicon - tertiary carbon",
        "[CX4H2]-[SiX4]-[CX4H1]",
        "CC[SiH2]C(C)C",
        "Silane silicon bonded to secondary and tertiary carbons.",
    ),
    EnvironmentRule(
        "secondary carbon - silane silicon - quaternary carbon",
        "[CX4H2]-[SiX4]-[CX4H0]",
        "CC[SiH2]C(C)(C)C",
        "Silane silicon bonded to secondary and quaternary carbons.",
    ),
    EnvironmentRule(
        "tertiary carbon - silane silicon - tertiary carbon",
        "[CX4H1]-[SiX4]-[CX4H1]",
        "CC(C)[SiH2]C(C)C",
        "Silane silicon bonded to tertiary and tertiary carbons.",
    ),
    EnvironmentRule(
        "tertiary carbon - silane silicon - quaternary carbon",
        "[CX4H1]-[SiX4]-[CX4H0]",
        "CC(C)[SiH2]C(C)(C)C",
        "Silane silicon bonded to tertiary and quaternary carbons.",
    ),
    EnvironmentRule(
        "quaternary carbon - silane silicon - quaternary carbon",
        "[CX4H0]-[SiX4]-[CX4H0]",
        "CC(C)(C)[SiH2]C(C)(C)C",
        "Silane silicon bonded to quaternary and quaternary carbons.",
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
    # --- Extended element coverage (see ENVIRONMENTS above) ----------
    BalanceSpecies("Water", "O", "Simplest oxygen balance species"),
    BalanceSpecies("Methanol", "CO", "Simplest alcohol"),
    BalanceSpecies("Ethanol", "CCO", "Two-carbon alcohol"),
    BalanceSpecies("Isopropanol", "CC(C)O", "Branched alcohol"),
    BalanceSpecies("tert-Butanol", "CC(C)(C)O", "Highly branched alcohol"),
    BalanceSpecies("Hydrogen sulfide", "S", "Simplest sulfur balance species"),
    BalanceSpecies("Methanethiol", "CS", "Simplest thiol"),
    BalanceSpecies("Dimethyl sulfide", "CSC", "Simplest thioether"),
    BalanceSpecies("Ethyl methyl sulfide", "CCSC", "Asymmetric thioether"),
    BalanceSpecies("Diethyl sulfide", "CCSCC", "Symmetric thioether"),
    BalanceSpecies("Isopropyl methyl sulfide", "CSC(C)C", "Branched thioether"),
    BalanceSpecies(
        "tert-Butyl methyl sulfide", "CSC(C)(C)C", "Highly branched thioether"
    ),
    BalanceSpecies("Dimethyl sulfoxide", "CS(=O)C", "Simplest sulfoxide"),
    BalanceSpecies("Dimethyl sulfone", "CS(=O)(=O)C", "Simplest sulfone"),
    BalanceSpecies("Phosphine", "P", "Simplest phosphorus balance species"),
    BalanceSpecies("Methylphosphine", "CP", "Primary phosphine"),
    BalanceSpecies("Dimethylphosphine", "CPC", "Secondary phosphine"),
    BalanceSpecies("Trimethylphosphine", "CP(C)C", "Tertiary phosphine"),
    BalanceSpecies("Fluoromethane", "CF", "Simplest fluoroalkane"),
    BalanceSpecies("Chloromethane", "CCl", "Simplest chloroalkane"),
    BalanceSpecies("Bromomethane", "CBr", "Simplest bromoalkane"),
    BalanceSpecies("Iodomethane", "CI", "Simplest iodoalkane"),
    BalanceSpecies("Fluoroethane", "CCF", "Two-carbon fluoroalkane"),
    BalanceSpecies("Chloroethane", "CCCl", "Two-carbon chloroalkane"),
    BalanceSpecies("Bromoethane", "CCBr", "Two-carbon bromoalkane"),
    BalanceSpecies("Iodoethane", "CCI", "Two-carbon iodoalkane"),
    BalanceSpecies("Silane", "[SiH4]", "Simplest silicon balance species"),
    BalanceSpecies("Methylsilane", "C[SiH3]", "Simplest organosilane"),
    BalanceSpecies("Tetramethylsilane", "C[Si](C)(C)C", "Fully methylated silane"),
    BalanceSpecies("Trimethylborane", "CB(C)C", "Simplest trialkylborane"),
    BalanceSpecies("Dimethyl selenide", "C[Se]C", "Simplest selenoether"),
    BalanceSpecies("Furan", "c1ccoc1", "Aromatic oxygen heterocycle"),
    BalanceSpecies("Thiophene", "c1ccsc1", "Aromatic sulfur heterocycle"),
    BalanceSpecies("Pyrrole", "c1cc[nH]c1", "Aromatic N-H heterocycle"),
    BalanceSpecies("Pyridine", "c1ccncc1", "Aromatic nitrogen heterocycle"),
    BalanceSpecies(
        "Dimethylsilane", "C[SiH2]C", "Silicon bridging two primary carbons"
    ),
    BalanceSpecies("Ethyl methyl silane", "CC[SiH2]C", "Asymmetric dialkylsilane"),
    BalanceSpecies("Diethylsilane", "CC[SiH2]CC", "Symmetric dialkylsilane"),
    BalanceSpecies("Isopropyl methyl silane", "C[SiH2]C(C)C", "Branched dialkylsilane"),
    BalanceSpecies(
        "tert-Butyl methyl silane", "C[SiH2]C(C)(C)C", "Highly branched dialkylsilane"
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
