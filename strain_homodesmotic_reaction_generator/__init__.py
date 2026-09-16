#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Strain Homodesmotic Reaction Generator plugin entry point.
"""

from typing import Any
from .ui import (
    open_analyzer_dialog,
    reset_session_state,
    restore_session_state,
    session_state,
)

PLUGIN_NAME = "Strain Homodesmotic Reaction Generator"
PLUGIN_VERSION = "0.8.0"
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
    # A substituted reference is part of how the strain energy was defined, so
    # it belongs in the project file rather than being retyped every session.
    if hasattr(context, "register_save_handler"):
        context.register_save_handler(session_state)
    if hasattr(context, "register_load_handler"):
        context.register_load_handler(restore_session_state)
    if hasattr(context, "register_document_reset_handler"):
        context.register_document_reset_handler(reset_session_state)
    if hasattr(context, "show_status_message"):
        context.show_status_message("Homodesmotic Reaction loaded.", 3000)
