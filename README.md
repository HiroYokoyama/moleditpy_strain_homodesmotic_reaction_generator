# Caged Molecule Homodesmotic Reaction Builder

This folder contains a MoleditPy plugin that detects common local bonding
environments in caged or highly constrained molecules and proposes small
reference molecules for draft homodesmotic or isodesmic reaction balancing.

Version: `0.1.0`

## Files

- `caged_homodesmotic_reaction_builder.py` - plugin entry point.
- `tests/test_analysis.py` - lightweight tests for the analysis/export logic.

## Install

Copy or symlink this folder into a MoleditPy plugin location, or load
`caged_homodesmotic_reaction_builder.py` with the plugin loader. The plugin
registers itself in the Analysis menu as
**Caged Molecule Homodesmotic Reaction Builder**.

## Usage

1. Open or draw a molecule in MoleditPy.
2. Choose **Analysis > Caged Molecule Homodesmotic Reaction Builder**.
3. Click **Analyze Current Molecule**.
4. Review the detected environments, automatically generated balancing species,
   and any unresolved atom-balance entries.
5. Export the analysis as CSV, HTML, or text if needed.

HTML export preserves the dialog color coding:

- Blue: original target molecule.
- Green: detected reference molecules.
- Yellow/purple: automatically added balancing species.
- Red: unresolved species that still need manual chemistry review.

The generated equation is a draft. The atom-balance summary shows which atoms
were balanced automatically and which atoms still need additional balancing
species before using quantum-chemical energies.

## Development Check

From this folder:

```bash
python -m pytest tests -v
```

The tests require RDKit. The GUI requires PyQt6.
