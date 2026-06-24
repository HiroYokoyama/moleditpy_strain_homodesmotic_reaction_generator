#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strain Homodesmotic Reaction Generator plugin entry point.
"""

from typing import Any
from .ui import open_analyzer_dialog

PLUGIN_NAME = "Strain Homodesmotic Reaction Generator"
PLUGIN_VERSION = "0.6.0"
PLUGIN_AUTHOR = "HiroYokoyama"
PLUGIN_DEPENDENCIES = ("numpy", "scipy")
PLUGIN_DESCRIPTION = (
    "Detect strain-molecule bonding environments and build a draft homodesmotic "
    "reaction balance."
)
PLUGIN_SUPPORTED_MOLEDITPY_VERSION = ">=3.0.0, <5.0.0"


def initialize(context: Any) -> None:
    """Initialize the plugin inside MoleditPy."""
    context.add_analysis_tool(
        "Homodesmotic Reaction", lambda: open_analyzer_dialog(context)
    )
    if hasattr(context, "show_status_message"):
        context.show_status_message("Homodesmotic Reaction loaded.", 3000)
