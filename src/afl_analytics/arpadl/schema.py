"""Schema for AR-PADL actions."""

from typing import Any, Optional

import pandera as pa
from pandera.typing import Series

from . import config as arpadlconfig


class ARPADLSchema(pa.DataFrameModel):
    """Definition of a SPADL dataframe."""

    match_id: Series[Any] = pa.Field()
    period_id: Series[int] = pa.Field(ge=1, le=4)
    time_seconds: Series[float] = pa.Field(ge=0)
    team: Series[Any] = pa.Field()
    player: Series[Any] = pa.Field()
    start_x: Series[float] = pa.Field()
    start_y: Series[float] = pa.Field()
    end_x: Series[float] = pa.Field()
    end_y: Series[float] = pa.Field()
    action_type: Optional[Series[str]] = pa.Field(isin=arpadlconfig.actiontypes_df().type_name)
    bodypart: Optional[Series[str]] = pa.Field(isin=arpadlconfig.bodyparts_df().bodypart_name)
    result: Optional[Series[str]] = pa.Field(isin=arpadlconfig.results_df().result_name)

    class Config:  # noqa: D106
        strict = True
        coerce = True