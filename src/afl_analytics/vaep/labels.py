"""Implements the label tranformers of the VAEP framework."""

import pandas as pd  # type: ignore
from pandera.typing import DataFrame

import afl_analytics.arpadl.config as spadl
from afl_analytics.arpadl.schema import ARPADLSchema


def scores(actions: DataFrame[ARPADLSchema], nr_actions: int = 10) -> pd.DataFrame:
    """Determine whether the team possessing the ball scored a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'scores' and a row for each action set to
        True if a goal was scored by the team possessing the ball within the
        next x actions; otherwise False.
    """
    # merging goals, owngoals and team_ids

    goals = actions["action_type"].str.contains("shot") & (
        actions["result"] == "goal"
    )
    y = pd.concat([goals, actions["team"]], axis=1)
    y.columns = ["goal", "team"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team", "goal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c].iloc[len(y) - 1]
            y["%s+%d" % (c, i)] = shifted
            
    res = y["goal"]
    for i in range(1, nr_actions):
        gi = y["goal+%d" % i] & (y["team+%d" % i] == y["team"])
        res = res | gi
    
    return pd.DataFrame(res, columns=["scores"])


def concedes(actions: DataFrame[ARPADLSchema], nr_actions: int = 10) -> pd.DataFrame:
    """Determine whether the team possessing the ball conceded a goal within the next x actions.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.
    nr_actions : int, default=10  # noqa: DAR103
        Number of actions after the current action to consider.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'concedes' and a row for each action set to
        True if a goal was conceded by the team possessing the ball within the
        next x actions; otherwise False.
    """
    goals = actions["action_type"].str.contains("shot") & (
        actions["result"] == "goal"
    )
    y = pd.concat([goals, actions["team"]], axis=1)
    y.columns = ["goal", "team"]

    # adding future results
    for i in range(1, nr_actions):
        for c in ["team", "goal"]:
            shifted = y[c].shift(-i)
            shifted[-i:] = y[c].iloc[len(y) - 1]
            y["%s+%d" % (c, i)] = shifted
            
    for i in range(1, nr_actions):
        gi = y["goal+%d" % i] & (y["team+%d" % i] != y["team"])
    
    return pd.DataFrame(gi, columns=["concedes"])


def goal_from_shot(actions: DataFrame[ARPADLSchema]) -> pd.DataFrame:
    """Determine whether a goal was scored from the current action.

    This label can be use to train an xG model.

    Parameters
    ----------
    actions : pd.DataFrame
        The actions of a game.

    Returns
    -------
    pd.DataFrame
        A dataframe with a column 'goal' and a row for each action set to
        True if a goal was scored from the current action; otherwise False.
    """
    goals = actions["action_type"].str.contains("shot") & (
        actions["result"] == "goal"
    )
    behinds = actions["action_type"].str.contains("shot") & (
        actions["result"] == "behind"
    )
    scores = goals*6 + behinds
    scores = pd.concat([goals, behinds, scores], axis=1)
    return pd.DataFrame(scores, columns=["goal_from_shot", "behind_from_shot", "score_from_shot"])