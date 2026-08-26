from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from scripts.generate_final_paper_tables import (
    CANONICAL_METRICS_PATH,
    CANONICAL_STATISTICS_PATH,
    MAIN_RESULTS_DISPLAY_COLUMNS,
    STATISTICS_DISPLAY_COLUMNS,
    generate_final_paper_tables,
)

pytestmark = pytest.mark.skipif(
    not CANONICAL_METRICS_PATH.exists() or not CANONICAL_STATISTICS_PATH.exists(),
    reason="canonical final_reproducible_run audit artifacts not present",
)


def test_generated_main_results_table_matches_canonical_metrics_audit_exactly() -> None:
    """No value in the generated main-results paper table may differ from the
    corresponding cell in final_canonical_metrics_audit.csv - this is the guard against
    a hand-typed or stale number ever entering a paper table."""
    tables = generate_final_paper_tables()
    canonical = pd.read_csv(CANONICAL_METRICS_PATH)
    generated = tables["main_results"]

    inverse_columns = {display: source for source, display in MAIN_RESULTS_DISPLAY_COLUMNS.items()}
    for display_col, source_col in inverse_columns.items():
        if pd.api.types.is_float_dtype(canonical[source_col]):
            pd.testing.assert_series_equal(
                generated[display_col].astype(float).round(4), canonical[source_col].round(4), check_names=False,
            )
        else:
            pd.testing.assert_series_equal(
                generated[display_col].astype(str), canonical[source_col].astype(str), check_names=False,
            )


def test_generated_statistical_comparison_table_matches_canonical_statistics_audit_exactly() -> None:
    tables = generate_final_paper_tables()
    canonical = pd.read_csv(CANONICAL_STATISTICS_PATH)
    generated = tables["statistical_comparison"]

    inverse_columns = {display: source for source, display in STATISTICS_DISPLAY_COLUMNS.items()}
    for display_col, source_col in inverse_columns.items():
        if pd.api.types.is_float_dtype(canonical[source_col]):
            pd.testing.assert_series_equal(
                generated[display_col].astype(float).round(4), canonical[source_col].round(4), check_names=False,
            )
        else:
            pd.testing.assert_series_equal(
                generated[display_col].astype(str), canonical[source_col].astype(str), check_names=False,
            )


def test_paper_tables_regenerate_deterministically() -> None:
    """Regenerating twice from the same frozen canonical sources must be byte-for-value
    identical - the generator must be a pure read/format/write, never a fresh
    computation that could drift between runs."""
    first = generate_final_paper_tables()
    second = generate_final_paper_tables()
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_generate_final_paper_tables_raises_if_canonical_source_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import scripts.generate_final_paper_tables as module

    monkeypatch.setattr(module, "CANONICAL_METRICS_PATH", tmp_path / "does_not_exist.csv")
    with pytest.raises(FileNotFoundError):
        module.generate_final_paper_tables()
