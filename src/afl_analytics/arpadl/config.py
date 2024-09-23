"""Configuration of the ARPADL language.

Attributes
----------
field_length : float
    The length of a pitch (in meters).
field_width : float
    The width of a pitch (in meters).
bodyparts : list(str)
    The bodyparts used in the SPADL language.
results : list(str)
    The action results used in the SPADL language.
actiontypes : list(str)
    The action types used in the SPADL language.

"""

import pandas as pd  # type: ignore

field_length: float = 165.0  # unit: meters
field_width: float = 135.0  # unit: meters

bodyparts: list[str] = ["foot", "hand"]

results: list[str] = ["fail", "success", "goal", "behind", "miss"]

actiontypes: list[str] = [
    'bounce',
    'carry',
    'error',
    'free',
    'free_off_ball',
    'gather',
    'gather_from_hitout',
    'gather_from_opposition',
    'handball',
    'hard_ball_get',
    'kick',
    'kick_ground',
    'kickin',
    'kickin_oof',
    'kickin_play_on',
    'knock_on',
    'loose_ball_get',
    'mark_contested',
    'mark_dropped',
    'mark_fumbled',
    'mark_on_lead',
    'mark_uncontested',
    'non_action',
    'shot',
    'spoil']


def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])


def results_df() -> pd.DataFrame:
    """Return a dataframe with the result id and result name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'result_id' and 'result_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(results)), columns=["result_id", "result_name"])


def bodyparts_df() -> pd.DataFrame:
    """Return a dataframe with the bodypart id and bodypart name of each SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'bodypart_id' and 'bodypart_name' of each SPADL action type.
    """
    return pd.DataFrame(list(enumerate(bodyparts)), columns=["bodypart_id", "bodypart_name"])