from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class CapacityState:
    remaining: dict[tuple[str, str], int] | None
    batch_column: str | None


def capacity_penalty_for_expert(
    case_row: pd.Series,
    expert_name: str,
    capacity_state: CapacityState,
    penalty_weight: float,
) -> float:
    if capacity_state.remaining is None or capacity_state.batch_column is None:
        return 0.0
    batch_value = str(case_row[capacity_state.batch_column])
    remaining = capacity_state.remaining.get((batch_value, expert_name), 0)
    if remaining <= 0:
        return penalty_weight
    return penalty_weight / float(remaining)


def allocate_expert(
    case_row: pd.Series,
    expert_name: str,
    capacity_state: CapacityState,
) -> bool:
    if capacity_state.remaining is None or capacity_state.batch_column is None:
        return True
    batch_value = str(case_row[capacity_state.batch_column])
    key = (batch_value, expert_name)
    remaining = capacity_state.remaining.get(key, 0)
    if remaining <= 0:
        return False
    capacity_state.remaining[key] = remaining - 1
    return True
