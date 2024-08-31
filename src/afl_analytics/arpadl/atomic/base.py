"""Implements a converter for regular SPADL actions to atomic actions."""

from typing import cast

import pandas as pd
from pandera.typing import DataFrame

import afl_analytics.arpadl.config as _spadl
from afl_analytics.arpadl.schema import ARPADLSchema

from . import config as _atomicspadl
from .schema import AtomicARPADLSchema


def convert_to_atomic(actions: DataFrame[ARPADLSchema]) -> DataFrame[AtomicARPADLSchema]:
    """Convert regular ARPADL actions to atomic actions.

    Parameters
    ----------
    actions : pd.DataFrame
        An ARPADL dataframe.

    Returns
    -------
    pd.DataFrame
        The Atomic-ARPADL dataframe.
    """
    atomic_actions = cast(pd.DataFrame, actions.copy())
    atomic_actions = _extra_from_disposals(atomic_actions)
    atomic_actions = _extra_from_shots(atomic_actions)
    atomic_actions = _extra_from_fouls(atomic_actions)
    atomic_actions = _convert_columns(atomic_actions)
    return cast(DataFrame[AtomicARPADLSchema], atomic_actions)


def _extra_from_disposals(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = actions.shift(-1)
    same_team = actions['team'] == next_actions['team']

    samegame = actions['match_id'] == next_actions['match_id']
    sameperiod = actions['period_id'] == next_actions['period_id']
    successful = actions['result'] == "success"
    handball = actions['action_type'] == "handball"
    # samephase = next_actions.time_seconds - actions.time_seconds < max_pass_duration
    extra_idx = (
        actions['action_type'].isin(["handball"])
        & samegame
        & sameperiod
        )

    prev = actions[extra_idx]
    nex = next_actions[extra_idx]

    extras = pd.DataFrame()
    prev = actions[extra_idx]
    nex = next_actions[extra_idx]
    extras["match_id"] = nex['match_id']
    extras["period_id"] = nex['period_id']
    extras["time_seconds"] = prev["time_seconds"]

    extras["team"] = nex['team']
    extras["player"] = nex['player']
    extras["start_x"] = prev['end_x']
    extras["start_y"] = prev['end_y']
    extras["end_x"] = nex['start_x']
    extras["end_y"] = nex['start_y']
    extras["bodypart"] = 'hand'

    extras['action_type'] = "non_action"
    extras['action_type'] = (
        extras['action_type'].mask(same_team & successful & handball, "handball_receival")
    )
    extras = extras[extras['action_type'] != 'non_action']

    extras['result'] = "success"

    actions = pd.concat([actions, extras], ignore_index=True, sort=False)
    actions = actions.sort_values(["match_id", "period_id", 'time_seconds']).reset_index(drop=True)
    return actions

def _extra_from_shots(actions: pd.DataFrame) -> pd.DataFrame:

    shotlike = ["shot"]

    shot = actions['action_type'].isin(shotlike)
    shot_goal = shot & (actions['result'] == "goal")
    shot_behind = shot & (actions['result'] == "behind")
    shot_miss = shot & (actions['result'] == "miss")

    extra_idx = shot_goal | shot_behind | shot_miss
    prev = actions[extra_idx]

    extras = pd.DataFrame()
    extras["match_id"] = prev['match_id']
    extras["period_id"] = prev['period_id']
    extras["time_seconds"] = prev["time_seconds"]

    extras["team"] = prev['team']
    extras["player"] = prev['player']
    extras["start_x"] = prev['end_x']
    extras["start_y"] = prev['end_y']
    extras["end_x"] = prev['start_x']
    extras["end_y"] = prev['start_y']
    extras["bodypart"] = 'foot'

    extras["action_type"] = "miss"
    extras["action_type"] = (
        extras['action_type'].mask(shot_goal, "goal")
        .mask(shot_behind, "behind")
        .mask(shot_miss, "miss")
    )

    actions = pd.concat([actions, extras], ignore_index=True, sort=False)
    actions = actions.sort_values(["match_id", "period_id", 'time_seconds']).reset_index(drop=True)
    return actions

def _extra_from_fouls(actions: pd.DataFrame) -> pd.DataFrame:
    next_actions = actions.shift(-1)
    same_team = actions['team'] == next_actions['team']
    same_game = actions['match_id'] == next_actions['match_id']
    same_period = actions['period_id'] == next_actions['period_id']

    free = actions['action_type'] == "free"

    dx = actions['end_x'] - next_actions['start_x']
    dy = actions['end_y'] - next_actions['start_y']
    far_enough = dx**2 + dy**2 >= 50**2

    fifty_idx = (
        same_team
        & same_game
        & same_period
        & far_enough
        & free
    )
    prev = actions[free]
    nex = next_actions[fifty_idx]

    extras = pd.DataFrame()
    extras["match_id"] = nex['match_id']
    extras["period_id"] = nex['period_id']
    extras["time_seconds"] = prev["time_seconds"]

    extras["team"] = nex['team']
    extras["player"] = nex['player']
    extras["start_x"] = prev['end_x']
    extras["start_y"] = prev['end_y']
    extras["end_x"] = nex['start_x']
    extras["end_y"] = nex['start_y']
    extras["bodypart"] = 'hand'
    extras["action_type"] = '50m_penalty'
    extras["result"] = 'success'

    actions = pd.concat([actions, extras], ignore_index=True, sort=False)
    actions = actions.sort_values(["match_id", "period_id", 'time_seconds']).reset_index(drop=True)
    return actions

def _convert_columns(actions: pd.DataFrame) -> pd.DataFrame:
    actions["x"] = actions['start_x']
    actions["y"] = actions['start_y']
    actions["dx"] = actions['end_x'] - actions['start_x']
    actions["dy"] = actions['end_y'] - actions['start_y']
    return actions[
        ['match_id',
        'period_id',
        'time_seconds',
        'team',
        'player',
        'x',
        'y',
        'dx',
        'dy',
        'action_type',
        'bodypart',
        ]
    ]


