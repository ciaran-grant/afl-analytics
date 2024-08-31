"""Configuration of the Atomic-SPADL language.

Attributes
----------
field_length : float
    The length of a pitch (in meters).
field_width : float
    The width of a pitch (in meters).
bodyparts : list(str)
    The bodyparts used in the Atomic-ARPADL language.
actiontypes : list(str)
    The action types used in the Atomic-ARPADL language.

"""

import pandas as pd

import afl_analytics.arpadl.config as _arpadl

field_length = _arpadl.field_length
field_width = _arpadl.field_width

bodyparts = _arpadl.bodyparts
bodyparts_df = _arpadl.bodyparts_df

actiontypes = _arpadl.actiontypes + [
    "handball_receive",
    "50m_penalty"
]


def actiontypes_df() -> pd.DataFrame:
    """Return a dataframe with the type id and type name of each Atomic-SPADL action type.

    Returns
    -------
    pd.DataFrame
        The 'type_id' and 'type_name' of each Atomic-SPADL action type.
    """
    return pd.DataFrame(list(enumerate(actiontypes)), columns=["type_id", "type_name"])