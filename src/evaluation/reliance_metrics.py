from __future__ import annotations

import pandas as pd


def _safe_mean(mask: pd.Series) -> float:
    if len(mask) == 0:
        return 0.0
    return float(mask.mean())


def compute_reliance_metrics(frame: pd.DataFrame, *, require_wrong_confident_label: bool = False) -> dict[str, float]:
    """Reliance metrics over a routed-decisions frame.

    Definitions (all denominators made explicit - see also
    docs/... conversation record for the full audit):

    - correct_rejection / overreliance / correct_reliance / underreliance (the
      DEFAULT, UNCONDITIONAL definitions returned under these exact keys, UNCHANGED
      from the original implementation): each is P(condition) over ALL N rows in
      `frame`, e.g. overreliance = mean_N[used_ai & ~ai_correct]. These keys are kept
      exactly as before - several optimization/constraint call sites (MAGD-Learned/
      Constrained weight search, MAGD-Constrained's feasibility check) consume them
      by name, so changing their meaning would silently alter already-frozen v1
      routing/threshold selection. They are also exposed under the explicit alias
      names `correct_rejection_all_cases` / `overreliance_all_cases` for clarity.
    - correct_rejection_given_ai_wrong / overreliance_given_ai_wrong (NEW, additive):
      the manuscript-preferred CONDITIONAL definitions - P(non-AI route | AI wrong)
      and P(AI route | AI wrong) respectively, i.e. denominator = count of AI-wrong
      rows only, not N. These two sum to exactly 1 whenever at least one AI-wrong row
      exists (they partition the AI-wrong subset by route taken).
    - wrong_confident_avoidance_rate (WCA): P(non-AI route | offline wrong-confident
      label == 1) - denominator = count of offline-labeled wrong-confident rows, not
      N. Requires `wrong_confident_label_offline` in `frame`. If
      `require_wrong_confident_label=True` and the column is absent, this raises
      instead of silently returning 0.0 - the previous silent fallback made several
      independently-computed WCA values (including an earlier draft L2D-Standard
      report) read as 0.0 for no reason other than a missing merge, not because the
      method genuinely never avoided a wrong-confident case. Default is False to
      preserve exact prior behaviour for existing callers that do not pass this label
      (MAGD-Constrained's feasibility search, MAGD-Learned/Constrained weight
      learning, MAGD-ValidationTuned's budget search) - changing those call sites'
      behaviour is out of scope here since wrong_confident_avoidance directly gates
      MAGD-Constrained's constraint-feasibility determination.
    """
    uses_ai = frame["used_ai"].astype(bool)
    ai_correct = frame["ai_correct"].astype(bool)
    ai_wrong = ~ai_correct

    correct_reliance = _safe_mean(uses_ai & ai_correct)
    correct_rejection = _safe_mean((~uses_ai) & (~ai_correct))
    overreliance = _safe_mean(uses_ai & (~ai_correct))
    underreliance = _safe_mean((~uses_ai) & ai_correct)

    if ai_wrong.any():
        correct_rejection_given_ai_wrong = float((~uses_ai)[ai_wrong].mean())
        overreliance_given_ai_wrong = float(uses_ai[ai_wrong].mean())
    else:
        correct_rejection_given_ai_wrong = 0.0
        overreliance_given_ai_wrong = 0.0

    if require_wrong_confident_label and "wrong_confident_label_offline" not in frame.columns and "offline_wrong_confident_label" not in frame.columns:
        raise ValueError(
            "compute_reliance_metrics(require_wrong_confident_label=True) requires a "
            "`wrong_confident_label_offline` column to compute WCA. Merge it in from "
            "the frozen wrong_confident_risk.csv (test split) before calling this "
            "function - do not silently default to 0."
        )
    wrong_conf = frame.get(
        "wrong_confident_label_offline",
        frame.get("offline_wrong_confident_label", pd.Series([0] * len(frame), index=frame.index)),
    ).astype(int)

    wrong_confident_cases = wrong_conf == 1
    if wrong_confident_cases.any():
        wrong_confident_avoidance = float((~uses_ai[wrong_confident_cases]).mean())
    else:
        wrong_confident_avoidance = 0.0

    return {
        "correct_reliance": correct_reliance,
        "correct_rejection": correct_rejection,
        "overreliance": overreliance,
        "underreliance": underreliance,
        "correct_rejection_all_cases": correct_rejection,
        "overreliance_all_cases": overreliance,
        "correct_rejection_given_ai_wrong": correct_rejection_given_ai_wrong,
        "overreliance_given_ai_wrong": overreliance_given_ai_wrong,
        "wrong_confident_avoidance_rate": wrong_confident_avoidance,
    }
