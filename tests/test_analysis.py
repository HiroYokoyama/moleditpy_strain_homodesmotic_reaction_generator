from collections import Counter
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from strain_homodesmotic_reaction_generator.core import (  # noqa: E402
    Chem,
    analyze_molecule,
    build_balance_terms,
    export_analysis,
    format_counter,
)
from strain_homodesmotic_reaction_generator import PLUGIN_DEPENDENCIES, PLUGIN_VERSION



pytestmark = pytest.mark.skipif(Chem is None, reason="RDKit is not available")


def test_analyze_ethane_detects_primary_primary_bond():
    mol = Chem.MolFromSmiles("CC")

    result = analyze_molecule(mol)

    assert len(result.matches) == 1
    assert result.matches[0].name == "primary carbon - primary carbon"
    assert result.matches[0].count == 1
    assert result.matches[0].reference_smiles == "CC"
    assert result.atom_delta == Counter()
    assert "CC -> 1 CC" in result.equation_text


def test_analyze_dimethyl_ether_detects_internal_ether_oxygen():
    mol = Chem.MolFromSmiles("COC")

    result = analyze_molecule(mol)

    names = {match.name for match in result.matches}
    assert "primary carbon - ether oxygen - primary carbon" in names
    assert any(match.reference_smiles == "COC" for match in result.matches)
    assert "#8ab4f8" in result.equation_html


def test_terminal_heteroatoms_are_not_treated_as_cage_ether_environments():
    methanol = Chem.MolFromSmiles("CO")
    mol = Chem.MolFromSmiles("CN")

    methanol_result = analyze_molecule(methanol)
    methylamine_result = analyze_molecule(mol)

    assert "primary carbon - ether oxygen" not in {
        match.name for match in methanol_result.matches
    }
    assert not any("amine" in match.name for match in methylamine_result.matches)


def test_build_balance_terms_uses_simple_reference_species():
    terms, unresolved = build_balance_terms(Counter({"C": 2, "H": 6}))

    assert unresolved == Counter()
    assert [(term.name, term.smiles, term.count) for term in terms] == [
        ("ethane", "CC", 1)
    ]


def test_build_balance_terms_resolves_large_hydrocarbon_balance():
    terms, unresolved = build_balance_terms(Counter({"C": 88, "H": 224}))

    atom_total = Counter()
    for term in terms:
        mol = Chem.AddHs(Chem.MolFromSmiles(term.smiles))
        for atom in mol.GetAtoms():
            atom_total[atom.GetSymbol()] += term.count

    assert unresolved == Counter()
    assert atom_total == Counter({"C": 88, "H": 224})


def test_format_counter_is_stable_and_english():
    assert format_counter(Counter({"H": 6, "C": 2})) == "C: 2, H: 6"
    assert format_counter(Counter()) == "none"


def test_export_analysis_writes_csv(tmp_path):
    result = analyze_molecule(Chem.MolFromSmiles("CC"))
    output = tmp_path / "analysis.csv"

    export_analysis(output, result, current_file_path="my_test_molecule.pmeprj")

    text = output.read_text(encoding="utf-8")
    assert "Homodesmotic Reaction Report" in text
    assert "File,my_test_molecule.pmeprj" in text
    assert "Generated on," in text
    assert "Environment,Count,Reference SMILES,Description" in text
    assert "primary carbon - primary carbon,1,CC" in text
    assert "Left-side balancing species" in text


def test_export_analysis_writes_colored_html(tmp_path):
    # Dimethyl ether (no balance)
    result = analyze_molecule(Chem.MolFromSmiles("COC"))
    output = tmp_path / "analysis.html"

    export_analysis(output, result, current_file_path="my_test_ether.pmeprj")

    text = output.read_text(encoding="utf-8")
    assert "<!doctype html>" in text
    assert "File: my_test_ether.pmeprj" in text
    assert "Generated on:" in text
    assert f"v{PLUGIN_VERSION}" in text
    assert "row-ref" in text
    assert "<td>Ref</td>" in text
    assert "<td>primary carbon - ether oxygen - primary carbon</td>" in text

    # Cyclopropane (has left balance)
    result_cyclopropane = analyze_molecule(Chem.MolFromSmiles("C1CC1"))
    output_cyclopropane = tmp_path / "analysis_cyclopropane.html"
    export_analysis(output_cyclopropane, result_cyclopropane)
    text_cyclopropane = output_cyclopropane.read_text(encoding="utf-8")
    assert "<td>Ref</td>" in text_cyclopropane
    assert "<td>secondary carbon - secondary carbon</td>" in text_cyclopropane
    assert "<td>Left Balance</td>" in text_cyclopropane
    assert "<td>propane</td>" in text_cyclopropane
    assert "<td>Three-carbon saturated hydrocarbon balance species.</td>" in text_cyclopropane


def test_export_analysis_writes_txt(tmp_path):
    result = analyze_molecule(Chem.MolFromSmiles("CC"))
    output = tmp_path / "analysis.txt"

    export_analysis(output, result, current_file_path="my_test_ethane.pmeprj")

    text = output.read_text(encoding="utf-8")
    assert "Homodesmotic Reaction Report" in text
    assert "File: my_test_ethane.pmeprj" in text
    assert "Generated on:" in text
    assert "CC -> 1 CC" in text


def test_analyze_adamantanone_detects_carbonyls():
    mol = Chem.MolFromSmiles("O=C1C2CC3CC1CC(C2)C3")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Tertiary-Carbonyl" in names
    assert not result.is_elemental_balance


def test_analyze_methylamine_detects_primary_amine():
    mol = Chem.MolFromSmiles("CN")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Primary-Primary-Amine" in names


def test_analyze_unsolvable_molecule_triggers_elemental_balance():
    mol = Chem.MolFromSmiles("C1N2CN3CN1CN(C2)C3") # Hexamethylenetetramine
    result = analyze_molecule(mol)
    assert result.is_elemental_balance
    assert "Calculated in elemental balance mode" in result.equation_html


def test_analyze_trimethylamine_detects_tertiary_amine():
    mol = Chem.MolFromSmiles("CN(C)C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Primary-Tertiary-Amine" in names
    assert not result.is_elemental_balance


def test_analyze_butanone_detects_secondary_carbonyl():
    mol = Chem.MolFromSmiles("CCC(=O)C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Secondary-Carbonyl" in names
    assert not result.is_elemental_balance


def test_plugin_metadata():
    assert "numpy" in PLUGIN_DEPENDENCIES
    assert "scipy" in PLUGIN_DEPENDENCIES


def test_analyze_fallback_triggered_when_milp_is_missing(monkeypatch):
    import strain_homodesmotic_reaction_generator.core as core
    monkeypatch.setattr(core, "milp", None)
    mol = Chem.MolFromSmiles("COC")
    result = core.analyze_molecule(mol)
    assert result.is_elemental_balance
    assert "Calculated in elemental balance mode" in result.equation_html


def test_analyze_benzene_aromatic():
    mol = Chem.MolFromSmiles("c1ccccc1")
    result = analyze_molecule(mol)
    assert len(result.matches) == 1
    assert result.matches[0].name == "Aromatic-CH"
    assert result.matches[0].count == 6
    assert not result.is_elemental_balance
    assert "c1ccccc1 -> 1 c1ccccc1" in result.equation_text


def test_analyze_toluene_aromatic():
    mol = Chem.MolFromSmiles("Cc1ccccc1")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Aromatic-CH" in names
    assert "Aromatic-C-Carbon" in names
    assert not result.is_elemental_balance
    assert "cc1ccccc1 -> 1 cc1ccccc1" in result.equation_text.lower()


def test_analyze_naphthalene_aromatic():
    mol = Chem.MolFromSmiles("c1ccc2ccccc2c1")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Aromatic-CH" in names
    assert "Aromatic-fusion-carbon" in names
    assert not result.is_elemental_balance


def test_analyze_biphenyl_aromatic():
    mol = Chem.MolFromSmiles("c1ccccc1-c2ccccc2")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Aromatic-CH" in names
    assert "Aromatic-C-Aromatic" in names
    assert not result.is_elemental_balance
    assert "c1ccc(-c2ccccc2)cc1 -> 1 c1ccc(-c2ccccc2)cc1" in result.equation_text.lower()


def test_analyze_five_cpp_aromatic():
    mol = Chem.MolFromSmiles("c1cc2ccc1-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc-2cc1")
    result = analyze_molecule(mol)
    assert not result.is_elemental_balance
    assert format_counter(result.target_atoms) == "C: 30, H: 20"
    assert format_counter(result.reference_atoms) == "C: 60, H: 50"
    assert format_counter(result.atom_delta) == "C: 30, H: 30"
    
    names_counts = {match.name: match.count for match in result.matches}
    assert names_counts["Aromatic-CH"] == 20
    assert names_counts["Aromatic-C-Aromatic"] == 10
    
    left_names = {term.name: term.count for term in result.left_balance_terms if term.count > 0}
    assert left_names["Benzene"] == 5
    assert "ethane" not in left_names
    
    right_active = [term for term in result.right_balance_terms if term.count > 0]
    assert len(right_active) == 0


def test_analyze_acetylene_sp():
    mol = Chem.MolFromSmiles("C#C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Terminal-alkyne" in names
    assert not result.is_elemental_balance
    assert "C#C -> 1 C#C" in result.equation_text


def test_analyze_propyne_sp():
    mol = Chem.MolFromSmiles("CC#C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Substituted-alkyne" in names
    assert not result.is_elemental_balance
    assert "C#CC -> 1 CC#C" in result.equation_text


def test_analyze_acetonitrile_sp():
    mol = Chem.MolFromSmiles("CC#N")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Nitrile" in names
    assert not result.is_elemental_balance
    assert "CC#N -> 1 CC#N" in result.equation_text


def test_analyze_ethene_sp2():
    mol = Chem.MolFromSmiles("C=C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Secondary-alkene" in names
    assert not result.is_elemental_balance
    assert "C=C -> 1 C=C" in result.equation_text


def test_analyze_propene_sp2():
    mol = Chem.MolFromSmiles("CC=C")
    result = analyze_molecule(mol)
    names = {match.name for match in result.matches}
    assert "Tertiary-alkene" in names
    assert not result.is_elemental_balance
    assert "C=CC -> 1 CC=C" in result.equation_text


def test_analyze_cyclopropane_balance_species_colored():
    mol = Chem.MolFromSmiles("C1CC1")
    result = analyze_molecule(mol)
    assert not result.is_elemental_balance
    assert "C1CC1 + 3 CCC (propane) -> 3 CCCC" in result.equation_text
    
    # Check that propane (CCC) is colored atom-by-atom in HTML
    # The middle carbon should be yellow (#fde293), outer carbons should be green (#81c995)
    assert "color:#81c995" in result.equation_html
    assert "color:#fde293" in result.equation_html
    
    # Verify that the term "3 CCC" is constructed with mixed colors
    # Atom 1 (middle) has core_color (#fde293), Atom 0 & 2 have added_color (#81c995)
    # The SMILES string is "CCC", so the first C is green, second C is yellow, third C is green
    import re
    pattern = r'3\s+<span\s+style="color:#81c995;[^>]*>C</span><span\s+style="color:#fde293;[^>]*>C</span><span\s+style="color:#81c995;[^>]*>C</span>'
    assert re.search(pattern, result.equation_html) is not None

    # Verify that the term "3 CCCC" on the right side is colored:
    # C1 (green), C2 (yellow), C3 (blue), C4 (green)
    pattern_butane = r'3\s+<span\s+style="color:#81c995;[^>]*>C</span><span\s+style="color:#fde293;[^>]*>C</span><span\s+style="color:#8ab4f8;[^>]*>C</span><span\s+style="color:#81c995;[^>]*>C</span>'
    assert re.search(pattern_butane, result.equation_html) is not None


def test_bond_type_analysis():
    # Ethane should have 1 C-C single bond and 6 C-H single bonds
    mol = Chem.MolFromSmiles("CC")
    result = analyze_molecule(mol)
    assert result.target_bonds["C-C (single)"] == 1
    assert result.target_bonds["C-H (single)"] == 6
    # Perfectly balanced, so reaction bond delta should be empty (none)
    assert format_counter(result.reaction_bonds_delta) == "none"


def test_reaction_type_classification():
    # Ethane should be classified as Hyperhomodesmotic
    result_ethane = analyze_molecule(Chem.MolFromSmiles("CC"))
    assert result_ethane.reaction_type == "Hyperhomodesmotic"

    # [5]CPP should also be classified as Hyperhomodesmotic
    result_cpp = analyze_molecule(Chem.MolFromSmiles("c1cc2ccc1-c1ccc(cc1)-c1ccc(cc1)-c1ccc(cc1)-c1ccc-2cc1"))
    assert result_cpp.reaction_type == "Hyperhomodesmotic"

    # Hexamethylenetetramine has environment matching constraints, falling back to Elemental Balance
    result_hmta = analyze_molecule(Chem.MolFromSmiles("C1N2CN3CN1CN(C2)C3"))
    assert result_hmta.reaction_type == "Elemental Balance"

def test_reaction_type_classification_levels():
    from strain_homodesmotic_reaction_generator.core import (
        classify_reaction_type,
        BalanceTerm,
        ReferenceTerm
    )

    # 1. Unbalanced: mismatched atoms or unresolved atoms
    assert classify_reaction_type(
        "CC",
        (),
        (),
        (),
        Counter({"C": 1}),
        Counter()
    ) == "Unbalanced"

    # 2. Hyperhomodesmotic (balanced environment groups and hybridized bonds)
    assert classify_reaction_type(
        "CC",
        (),
        (ReferenceTerm(1, "CC", (0, 1)),),
        (),
        Counter(),
        Counter()
    ) == "Hyperhomodesmotic"

    # 3. Homodesmotic (balanced atoms and groups, but mismatched hybridized bonds)
    # e.g., 2 Propane -> Ethane + Butane
    assert classify_reaction_type(
        "CCC",
        (BalanceTerm("propane", "CCC", 1),),
        (),
        (BalanceTerm("ethane", "CC", 1), BalanceTerm("butane", "CCCC", 1)),
        Counter(),
        Counter()
    ) == "Homodesmotic"

    # 4. Isodesmic (balanced simple bonds, but mismatched hybridized bonds and groups)
    # e.g., Propene + Ethane -> Propane + Ethene
    assert classify_reaction_type(
        "CC=C",
        (BalanceTerm("ethane", "CC", 1),),
        (),
        (BalanceTerm("propane", "CCC", 1), BalanceTerm("Ethene", "C=C", 1)),
        Counter(),
        Counter()
    ) == "Isodesmic"

    # 5. Elemental Balance (only atoms match, not bonds)
    assert classify_reaction_type(
        "C=C",
        (BalanceTerm("methane", "C", 2),),
        (),
        (BalanceTerm("propane", "CCC", 1), BalanceTerm("methane", "C", 1)),
        Counter(),
        Counter()
    ) == "Elemental Balance"


