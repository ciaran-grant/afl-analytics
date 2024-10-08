
"""Implements the feature tranformers of the VAEP framework."""

from functools import wraps
from typing import Any, Callable, Union, no_type_check

import numpy as np  # type: ignore
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

import afl_analytics.arpadl.config as arpadlcfg
from afl_analytics.arpadl.atomic.schema import AtomicARPADLSchema
from afl_analytics.arpadl.schema import ARPADLSchema

arpadlActions = DataFrame[ARPADLSchema]
Actions = Union[DataFrame[ARPADLSchema], DataFrame[AtomicARPADLSchema]]
GameStates = list[Actions]
Features = DataFrame[Any]
FeatureTransfomer = Callable[[GameStates], Features]


def feature_column_names(fs: list[FeatureTransfomer], nb_prev_actions: int = 3) -> list[str]:
    """Return the names of the features generated by a list of transformers.

    Parameters
    ----------
    fs : list(callable)
        A list of feature transformers.
    nb_prev_actions : int, default=3  # noqa: DAR103
        The number of previous actions included in the game state.

    Returns
    -------
    list(str)
        The name of each generated feature.
    """
    arpadlcolumns = [
        'match_id',
        'period_id',
        'time_seconds',
        'team',
        'player',
        'start_x',
        'start_y',
        'end_x',
        'end_y',
        'action_type',
        'bodypart',
        'result'
    ]
    dummy_actions = pd.DataFrame(np.zeros((10, len(arpadlcolumns))), columns=arpadlcolumns)
    for c in arpadlcolumns:
        if "name" in c:
            dummy_actions[c] = dummy_actions[c].astype(str)
    gs = gamestates(dummy_actions, nb_prev_actions)  # type: ignore
    return list(pd.concat([f(gs) for f in fs], axis=1).columns.values)


def gamestates(actions: Actions, nb_prev_actions: int = 3) -> GameStates:
    r"""Convert a dataframe of actions to gamestates.

    Each gamestate is represented as the <nb_prev_actions> previous actions.

    The list of gamestates is internally represented as a list of actions
    dataframes :math:`[a_0,a_1,\ldots]` where each row in the a_i dataframe contains the
    previous action of the action in the same row in the :math:`a_{i-1}` dataframe.

    Parameters
    ----------
    actions : Actions
        A DataFrame with the actions of a game.
    nb_prev_actions : int, default=3  # noqa: DAR103
        The number of previous actions included in the game state.

    Raises
    ------
    ValueError
        If the number of actions is smaller 1.

    Returns
    -------
    GameStates
         The <nb_prev_actions> previous actions for each action.
    """
    if nb_prev_actions < 1:
        raise ValueError("The game state should include at least one preceding action.")
    states = [actions]
    for i in range(1, nb_prev_actions):
        prev_actions = actions.groupby(["match_id", "period_id"], sort=False, as_index=False).apply(
            lambda x: x.shift(i, fill_value=float("nan")).fillna(x.iloc[0])  # noqa: B023
        )
        prev_actions.index = actions.index.copy()
        states.append(prev_actions)  # type: ignore
    return states


def play_left_to_right(gamestates: GameStates, home_team: str) -> GameStates:
    """Perform all actions in a gamestate in the same playing direction.

    This changes the start and end location of each action in a gamestate,
    such that all actions are performed as if the team that performs the first
    action in the gamestate plays from left to right.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.
    home_team_id : int
        The ID of the home team.

    Returns
    -------
    GameStates
        The game states with all actions performed left to right.

    See Also
    --------
    socceraction.vaep.features.play_left_to_right : For transforming actions.
    """
    a0 = gamestates[0]
    away_idx = a0.team != home_team
    for actions in gamestates:
        for col in ["start_x", "end_x"]:
            actions.loc[away_idx, col] = arpadlcfg.field_length - actions[away_idx][col].values
        for col in ["start_y", "end_y"]:
            actions.loc[away_idx, col] = arpadlcfg.field_width - actions[away_idx][col].values
    return gamestates


@no_type_check
def simple(actionfn: Callable) -> FeatureTransfomer:
    """Make a function decorator to apply actionfeatures to game states.

    Parameters
    ----------
    actionfn : Callable
        A feature transformer that operates on actions.

    Returns
    -------
    FeatureTransfomer
        A feature transformer that operates on game states.
    """

    @wraps(actionfn)
    def _wrapper(gamestates: list[Actions]) -> pd.DataFrame:
        if not isinstance(gamestates, (list,)):
            gamestates = [gamestates]
        X = []
        for i, a in enumerate(gamestates):
            Xi = actionfn(a)
            Xi.columns = [c + "_a" + str(i) for c in Xi.columns]
            X.append(Xi)
        return pd.concat(X, axis=1)

    return _wrapper


# SIMPLE FEATURES


# @simple
# def actiontype(actions: Actions) -> Features:
#     """Get the type of each action.

#     Parameters
#     ----------
#     actions : Actions
#         The actions of a game.

#     Returns
#     -------
#     Features
#         The 'type_id' of each action.
#     """
#     X = pd.DataFrame(index=actions.index)
#     X["actiontype"] = pd.Categorical(
#         actions["type_id"].replace(arpadlcfg.actiontypes_df().type_name.to_dict()),
#         categories=arpadlcfg.actiontypes,
#         ordered=False,
#     )
#     return X


@simple
def actiontype_onehot(actions: arpadlActions) -> Features:
    """Get the one-hot-encoded type of each action.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        A one-hot encoding of each action's type.
    """
    X = {}
    for _, type_name in enumerate(arpadlcfg.actiontypes):
        col = "actiontype_" + type_name
        X[col] = actions["action_type"] == type_name
    return pd.DataFrame(X, index=actions.index)


# @simple
# def result(actions: arpadlActions) -> Features:
#     """Get the result of each action.

#     Parameters
#     ----------
#     actions : arpadlActions
#         The actions of a game.

#     Returns
#     -------
#     Features
#         The 'result_id' of each action.
#     """
#     X = pd.DataFrame(index=actions.index)
#     X["result"] = pd.Categorical(
#         actions["result_id"].replace(arpadlcfg.results_df().result_name.to_dict()),
#         categories=arpadlcfg.results,
#         ordered=False,
#     )
#     return X


@simple
def result_onehot(actions: arpadlActions) -> Features:
    """Get the one-hot-encode result of each action.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The one-hot encoding of each action's result.
    """
    X = {}
    for _, result_name in enumerate(arpadlcfg.results):
        col = "result_" + result_name
        X[col] = actions["result"] == result_name
    return pd.DataFrame(X, index=actions.index)


@simple
def actiontype_result_onehot(actions: arpadlActions) -> Features:
    """Get a one-hot encoding of the combination between the type and result of each action.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The one-hot encoding of each action's type and result.
    """
    res = result_onehot.__wrapped__(actions)  # type: ignore
    tys = actiontype_onehot.__wrapped__(actions)  # type: ignore
    df = {}
    for tyscol in list(tys.columns):
        for rescol in list(res.columns):
            df[tyscol + "_" + rescol] = tys[tyscol] & res[rescol]
    return pd.DataFrame(df, index=actions.index)


# @simple
# def bodypart(actions: Actions) -> Features:
#     """Get the body part used to perform each action.

#     This feature generator does not distinguish between the left and right foot.

#     Parameters
#     ----------
#     actions : Actions
#         The actions of a game.

#     Returns
#     -------
#     Features
#         The 'bodypart_id' of each action.

#     See Also
#     --------
#     bodypart_detailed :
#         An alternative version that splits between the left and right foot.
#     """
#     X = pd.DataFrame(index=actions.index)
#     foot_id = arpadlcfg.bodyparts.index("foot")
#     left_foot_id = arpadlcfg.bodyparts.index("foot_left")
#     right_foot_id = arpadlcfg.bodyparts.index("foot_right")
#     X["bodypart"] = pd.Categorical(
#         actions["bodypart_id"]
#         .replace([left_foot_id, right_foot_id], foot_id)
#         .replace(arpadlcfg.bodyparts_df().bodypart_name.to_dict()),
#         categories=["foot", "head", "other", "head/other"],
#         ordered=False,
#     )
#     return X


# @simple
# def bodypart_detailed(actions: Actions) -> Features:
#     """Get the body part with split by foot used to perform each action.

#     This feature generator distinguishes between the left and right foot, if
#     supported by the dataprovider.

#     Parameters
#     ----------
#     actions : Actions
#         The actions of a game.

#     Returns
#     -------
#     Features
#         The 'bodypart_id' of each action.

#     See Also
#     --------
#     bodypart :
#         An alternative version that does not split between the left and right foot.
#     """
#     X = pd.DataFrame(index=actions.index)
#     X["bodypart"] = pd.Categorical(
#         actions["bodypart_id"].replace(arpadlcfg.bodyparts_df().bodypart_name.to_dict()),
#         categories=arpadlcfg.bodyparts,
#         ordered=False,
#     )
#     return X


@simple
def bodypart_onehot(actions: Actions) -> Features:
    """Get the one-hot-encoded bodypart of each action.

    This feature generator does not distinguish between the left and right foot.

    Parameters
    ----------
    actions : Actions
        The actions of a game.

    Returns
    -------
    Features
        The one-hot encoding of each action's bodypart.

    See Also
    --------
    bodypart_detailed_onehot :
        An alternative version that splits between the left and right foot.
    """
    X = {}
    for _, bodypart_name in enumerate(arpadlcfg.bodyparts):
        # if bodypart_name in ("foot_left", "foot_right"):
        #     continue
        col = "bodypart_" + bodypart_name
        # if bodypart_name == "foot":
        #     foot_id = arpadlcfg.bodyparts.index("foot")
        #     left_foot_id = arpadlcfg.bodyparts.index("foot_left")
        #     right_foot_id = arpadlcfg.bodyparts.index("foot_right")
        #     X[col] = actions["bodypart_id"].isin([foot_id, left_foot_id, right_foot_id])
        # elif bodypart_name == "head/other":
        #     head_id = arpadlcfg.bodyparts.index("head")
        #     other_id = arpadlcfg.bodyparts.index("other")
        #     head_other_id = arpadlcfg.bodyparts.index("head/other")
        #     X[col] = actions["bodypart_id"].isin([head_id, other_id, head_other_id])
        # else:
        X[col] = actions["bodypart"] == bodypart_name
    return pd.DataFrame(X, index=actions.index)


# @simple
# def bodypart_detailed_onehot(actions: Actions) -> Features:
#     """Get the one-hot-encoded bodypart with split by foot of each action.

#     This feature generator distinguishes between the left and right foot, if
#     supported by the dataprovider.

#     Parameters
#     ----------
#     actions : Actions
#         The actions of a game.

#     Returns
#     -------
#     Features
#         The one-hot encoding of each action's bodypart.

#     See Also
#     --------
#     bodypart_onehot :
#         An alternative version that does not split between the left and right foot.
#     """
#     X = {}
#     for bodypart_id, bodypart_name in enumerate(arpadlcfg.bodyparts):
#         col = "bodypart_" + bodypart_name
#         if bodypart_name == "foot":
#             foot_id = arpadlcfg.bodyparts.index("foot")
#             left_foot_id = arpadlcfg.bodyparts.index("foot_left")
#             right_foot_id = arpadlcfg.bodyparts.index("foot_right")
#             X[col] = actions["bodypart_id"].isin([foot_id, left_foot_id, right_foot_id])
#         elif bodypart_name == "head/other":
#             head_id = arpadlcfg.bodyparts.index("head")
#             other_id = arpadlcfg.bodyparts.index("other")
#             head_other_id = arpadlcfg.bodyparts.index("head/other")
#             X[col] = actions["bodypart_id"].isin([head_id, other_id, head_other_id])
#         else:
#             X[col] = actions["bodypart_id"] == bodypart_id
#     return pd.DataFrame(X, index=actions.index)


# @simple
# def time(actions: Actions) -> Features:
#     """Get the time when each action was performed.

#     This generates the following features:
#         :period_id:
#             The ID of the period.
#         :time_seconds:
#             Seconds since the start of the period.
#         :time_seconds_overall:
#             Seconds since the start of the game. Stoppage time during previous
#             periods is ignored.

#     Parameters
#     ----------
#     actions : Actions
#         The actions of a game.

#     Returns
#     -------
#     Features
#         The 'period_id', 'time_seconds' and 'time_seconds_overall' when each
#         action was performed.
#     """
#     match_time_at_period_start = {1: 0, 2: 45, 3: 90, 4: 105, 5: 120}
#     timedf = actions[["period_id", "time_seconds"]].copy()
#     timedf["time_seconds_overall"] = (
#         timedf.period_id.map(match_time_at_period_start) * 60
#     ) + timedf.time_seconds
#     return timedf


@simple
def startlocation(actions: arpadlActions) -> Features:
    """Get the location where each action started.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The 'start_x' and 'start_y' location of each action.
    """
    return actions[["start_x", "start_y"]]


@simple
def endlocation(actions: arpadlActions) -> Features:
    """Get the location where each action ended.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The 'end_x' and 'end_y' location of each action.
    """
    return actions[["end_x", "end_y"]]


_goal_x: float = arpadlcfg.field_length
_goal_y: float = arpadlcfg.field_width / 2


@simple
def startpolar(actions: arpadlActions) -> Features:
    """Get the polar coordinates of each action's start location.

    The center of the opponent's goal is used as the origin.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The 'start_dist_to_goal' and 'start_angle_to_goal' of each action.
    """
    polardf = pd.DataFrame(index=actions.index)
    dx = (_goal_x - actions["start_x"]).abs().values
    dy = (_goal_y - actions["start_y"]).abs().values
    polardf["start_dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["start_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@simple
def endpolar(actions: arpadlActions) -> Features:
    """Get the polar coordinates of each action's end location.

    The center of the opponent's goal is used as the origin.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The 'end_dist_to_goal' and 'end_angle_to_goal' of each action.
    """
    polardf = pd.DataFrame(index=actions.index)
    dx = (_goal_x - actions["end_x"]).abs().values
    dy = (_goal_y - actions["end_y"]).abs().values
    polardf["end_dist_to_goal"] = np.sqrt(dx**2 + dy**2)
    with np.errstate(divide="ignore", invalid="ignore"):
        polardf["end_angle_to_goal"] = np.nan_to_num(np.arctan(dy / dx))
    return polardf


@simple
def movement(actions: arpadlActions) -> Features:
    """Get the distance covered by each action.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The horizontal ('dx'), vertical ('dy') and total ('movement') distance
        covered by each action.
    """
    mov = pd.DataFrame(index=actions.index)
    mov["dx"] = actions.end_x - actions.start_x
    mov["dy"] = actions.end_y - actions.start_y
    mov["movement"] = np.sqrt(mov.dx**2 + mov.dy**2)
    return mov


@simple
def player_possession_time(actions: arpadlActions) -> Features:
    """Get the time (sec) a player was in ball possession before attempting the action.

    We only look at the dribble preceding the action and reset the possession
    time after a defensive interception attempt or a take-on.

    Parameters
    ----------
    actions : arpadlActions
        The actions of a game.

    Returns
    -------
    Features
        The 'player_possession_time' of each action.
    """
    cur_action = actions[["period_id", "time_seconds", "player", "action_type"]]
    prev_action = actions.copy().shift(1)[["period_id", "time_seconds", "player", "action_type"]]
    df = cur_action.join(prev_action, rsuffix="_prev")
    same_player = df['player'] == df['player_prev']
    same_period = df['period_id'] == df['period_id_prev']
    prev_carry = df['action_type_prev'] == arpadlcfg.actiontypes.index("carry")
    mask = same_period & same_player & prev_carry
    df.loc[mask, "player_possession_time"] = (
        df.loc[mask, "time_seconds"] - df.loc[mask, "time_seconds_prev"]
    )
    return df[["player_possession_time"]].fillna(0.0)


# STATE FEATURES


def team(gamestates: GameStates) -> Features:
    """Check whether the possession changed during the game state.

    For each action in the game state, True if the team that performed the
    action is the same team that performed the last action of the game state;
    otherwise False.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with a column 'team_ai' for each <nb_prev_actions> indicating
        whether the team that performed action a0 is in possession.
    """
    a0 = gamestates[0]
    teamdf = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        teamdf["team_" + (str(i + 1))] = a['team'] == a0['team']
    return teamdf


def time_delta(gamestates: GameStates) -> Features:
    """Get the number of seconds between the last and previous actions.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with a column 'time_delta_i' for each <nb_prev_actions>
        containing the number of seconds between action ai and action a0.
    """
    a0 = gamestates[0]
    dt = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dt["time_delta_" + (str(i + 1))] = a0['time_seconds'] - a['time_seconds']
    return dt


def space_delta(gamestates: GameStates) -> Features:
    """Get the distance covered between the last and previous actions.

    Parameters
    ----------
    gamestates : GameStates
        The gamestates of a game.

    Returns
    -------
    Features
        A dataframe with a column for the horizontal ('dx_a0i'), vertical
        ('dy_a0i') and total ('mov_a0i') distance covered between each
        <nb_prev_actions> action ai and action a0.
    """
    a0 = gamestates[0]
    spaced = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a0.start_x
        spaced["dx_a0" + (str(i + 1))] = dx
        dy = a.end_y - a0.start_y
        spaced["dy_a0" + (str(i + 1))] = dy
        spaced["mov_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2)
    return spaced


def speed(gamestates: GameStates) -> Features:
    """Get the speed at which the ball moved during the previous actions.

    Parameters
    ----------
    gamestates : GameStates
        The game states of a game.

    Returns
    -------
    Features
        A dataframe with columns 'speedx_a0i', 'speedy_a0i', 'speed_a0i'
        for each <nb_prev_actions> containing the ball speed in m/s  between
        action ai and action a0.
    """
    a0 = gamestates[0]
    speed = pd.DataFrame(index=a0.index)
    for i, a in enumerate(gamestates[1:]):
        dx = a.end_x - a0.start_x
        dy = a.end_y - a0.start_y
        dt = a0.time_seconds - a.time_seconds
        dt[dt <= 0] = 1e-6
        speed["speedx_a0" + (str(i + 1))] = dx.abs() / dt
        speed["speedy_a0" + (str(i + 1))] = dy.abs() / dt
        speed["speed_a0" + (str(i + 1))] = np.sqrt(dx**2 + dy**2) / dt
    return speed


# CONTEXT FEATURES


def goalscore(gamestates: GameStates) -> Features:
    """Get the number of goals scored by each team after the action.

    Parameters
    ----------
    gamestates : GameStates
        The gamestates of a game.

    Returns
    -------
    Features
        The number of goals scored by the team performing the last action of the
        game state ('goalscore_team'), by the opponent ('goalscore_opponent'),
        and the goal difference between both teams ('goalscore_diff').
    """
    actions = gamestates[0]
    teamA = actions["team"].values[0]
    goals = actions["action_type"].str.contains("shot") & (
        actions["result"] == "goal"
    )
    teamisA = actions["team"] == teamA
    teamisB = ~teamisA
    goalsteamA = (goals & teamisA)
    goalsteamB = (goals & teamisB)
    goalscoreteamA = goalsteamA.cumsum() - goalsteamA
    goalscoreteamB = goalsteamB.cumsum() - goalsteamB

    scoredf = pd.DataFrame(index=actions.index)
    scoredf["goalscore_team"] = (goalscoreteamA * teamisA) + (goalscoreteamB * teamisB)
    scoredf["goalscore_opponent"] = (goalscoreteamB * teamisA) + (goalscoreteamA * teamisB)
    scoredf["goalscore_diff"] = scoredf["goalscore_team"] - scoredf["goalscore_opponent"]
    return scoredf
