# SPDX-License-Identifier: Apache-2.0
"""Locate the AlpaSim checkout the plugin runs against."""
from __future__ import annotations

import os
from pathlib import Path


def find_alpasim_root() -> Path:
    """Return the AlpaSim checkout root.

    Order: $ALPASIM_ROOT, any parent of this file (the plugin deployed under
    <root>/plugins/), then the pinned submodule at <repo>/third_party/alpasim.
    """
    here = Path(__file__).resolve()
    candidates = [Path(os.environ["ALPASIM_ROOT"])] if os.environ.get("ALPASIM_ROOT") else []
    candidates += list(here.parents)
    candidates.append(here.parents[2] / "third_party" / "alpasim")
    for root in candidates:
        if (root / "src" / "wizard").is_dir() and (root / "pyproject.toml").is_file():
            return root.resolve()
    raise RuntimeError(
        "Cannot find an AlpaSim checkout. Set ALPASIM_ROOT to one, deploy the plugin "
        "under <alpasim>/plugins/ (see sync.sh), or run "
        "'git submodule update --init third_party/alpasim'."
    )
