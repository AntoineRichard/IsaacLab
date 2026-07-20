# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for locked benchmark environment validation."""

from pathlib import Path

import pytest

from benchmarks.kamino_dvi.environment import EnvironmentProvenance, GitState, python_executable, validate_environment
from benchmarks.kamino_dvi.matrix import DEFAULT_MATRIX_PATH, load_matrix
from benchmarks.kamino_dvi.models import EnvironmentLabel


@pytest.fixture
def matrix():
    return load_matrix(DEFAULT_MATRIX_PATH)


@pytest.fixture
def provenance(matrix):
    return EnvironmentProvenance(
        python=Path("/repo/.venv-current/bin/python"),
        packages={"newton": "0.1.0", "torch": "2.11.0"},
        newton_path=Path("/venv/site-packages/newton/__init__.py"),
        newton_revision=matrix.revisions.newton_current,
        isaaclab=GitState(
            head="059e5e8d8a700000000000000000000000000000",
            ancestors=frozenset({matrix.revisions.isaaclab, matrix.revisions.schema}),
            dirty=False,
        ),
    )


def test_python_executable_maps_locked_environment_labels():
    """Environment labels must resolve to worktree-local interpreters."""
    root = Path("/repo")
    assert python_executable(root, EnvironmentLabel.CURRENT) == root / ".venv-current/bin/python"
    assert python_executable(root, EnvironmentLabel.PR3570) == root / ".venv-pr3570/bin/python"


@pytest.mark.parametrize("label", [EnvironmentLabel.CURRENT, EnvironmentLabel.PR3570])
def test_environment_accepts_required_isaaclab_ancestry_and_newton_revision(matrix, provenance, label):
    """A run may start only from the approved IsaacLab lineage and Newton commit."""
    newton_revision = (
        matrix.revisions.newton_current if label is EnvironmentLabel.CURRENT else matrix.revisions.newton_pr
    )
    candidate = provenance.replace(newton_revision=newton_revision)

    validate_environment(matrix, label, candidate)


def test_environment_rejects_wrong_newton_revision(matrix, provenance):
    """A mislabeled Newton installation must fail before training."""
    candidate = provenance.replace(newton_revision="0" * 40)

    with pytest.raises(ValueError, match="Newton revision"):
        validate_environment(matrix, EnvironmentLabel.CURRENT, candidate)


def test_environment_rejects_missing_isaaclab_base_revision(matrix, provenance):
    """The benchmark branch must descend from the frozen IsaacLab base."""
    candidate = provenance.replace(
        isaaclab=GitState(head=provenance.isaaclab.head, ancestors=frozenset({matrix.revisions.schema}), dirty=False)
    )

    with pytest.raises(ValueError, match="IsaacLab base revision"):
        validate_environment(matrix, EnvironmentLabel.CURRENT, candidate)


def test_environment_rejects_missing_schema_revision(matrix, provenance):
    """The benchmark branch must contain the success-series schema prerequisite."""
    candidate = provenance.replace(
        isaaclab=GitState(head=provenance.isaaclab.head, ancestors=frozenset({matrix.revisions.isaaclab}), dirty=False)
    )

    with pytest.raises(ValueError, match="schema prerequisite"):
        validate_environment(matrix, EnvironmentLabel.CURRENT, candidate)
