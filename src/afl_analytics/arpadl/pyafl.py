"""PyAFL event stream data to ARPADL converter."""


import numpy as np
import pandas as pd  # type: ignore
from pandera.typing import DataFrame

from .schema import ARPADLSchema

def convert_to_actions(chains: pd.DataFrame) -> DataFrame[ARPADLSchema]:
    """
    Convert PyAFL events to ARPADL actions.

    Parameters
    ----------
    chains : pd.DataFrame
        DataFrame containing AFL API Match Chains from a single game.

    Returns
    -------
    actions : pd.DataFrame
        DataFrame with corresponding ARPADL actions.

    """
    
    actions = chains.copy()

    actions["match_id"] = actions['Match_ID']
    actions["period_id"] = actions['Period_Number']
    actions["team"] = actions['Team']
    actions["player"] = actions['Player']    
    actions['time_seconds'] = _create_time_seconds(actions)
    actions['action_type'] = _create_action_type(actions)
    actions['bodypart'] = _create_bodypart(actions)
    actions['result'] = _create_result(actions)
    actions['start_x'], actions['start_y'] = _create_start_location(actions)

    actions = _filter_action_type(actions)
    actions = _remove_missing_players(actions)

    actions['end_x'], actions['end_y'] = _create_end_location(actions)
    actions = _remove_duplicate_actions(actions)

    actions = _filter_arpadl_columns(actions)
    actions = actions[~(actions.duplicated())]

    actions = _add_carries(actions)

    actions = actions.sort_values(by = ['match_id', 'period_id', 'time_seconds'], ascending = True)
        
    return ARPADLSchema.validate(actions)

def _remove_missing_players(chains):
    
    chains = chains[~chains['Player'].isna()]
    
    return chains

def _filter_action_type(chains):
    
    chains = chains[chains['action_type'] != 'non_action']
    chains = chains[~chains['action_type'].isna()]
    
    return chains

description_to_action_mapping = {
    'Kick':'kick',
    'Handball':'handball',
    'Handball Received': 'non_action',
    'Uncontested Mark':'mark_uncontested',
    'Loose Ball Get':'loose_ball_get',
    'Kick Into F50':'non_action',
    'Kick Inside 50 Result':'non_action',
    'Spoil':'spoil',
    'Hard Ball Get':'hard_ball_get',
    'Loose Ball Get Crumb':'loose_ball_get',
    'Gather':'gather',
    'Out of Bounds':'non_action',
    'Free For':'free',
    'Contested Mark':'mark_contested',
    'Contest Target':'non_action',
    'Ball Up Call':'non_action',
    'Centre Bounce':'non_action',
    'Goal':'non_action',
    'Gather From Hitout':'gather_from_hitout',
    'Kickin play on':'kickin_play_on',
    'Behind':'non_action',
    'Contested Knock On':'knock_on',
    'Ground Kick':'kick_ground',
    'Mark On Lead':'mark_on_lead',
    'Hard Ball Get Crumb':'hard_ball_get',
    'Gather from Opposition':'gather_from_opposition',
    'Bounce':'bounce',
    'Mark Fumbled':'non_action', # same as mark_dropped mostly
    'Mark Dropped':'mark_dropped',
    'OOF Kick In':'kickin_oof',
    'Out On Full After Kick':'non_action',
    'Ruck Hard Ball Get':'hard_ball_get',
    'No Pressure Error':'error',
    'Free For: In Possession':'free',
    'Free Advantage':'non_action',
    'Kickin short':'kickin',
    'Knock On':'knock_on',
    'Free For: Off The Ball':'free_off_ball'
}

def _create_action_type(chains):
    
    action_type = chains['Description'].map(description_to_action_mapping)
    action_type = np.where((chains['Shot_At_Goal']=="TRUE") | (chains['Shot_At_Goal'] == True), 'shot', action_type)
      
    return action_type

def _create_time_seconds(chains):
    max_quarter_durations = chains.groupby(['Match_ID', "Period_Number"])['Period_Duration'].max().reset_index()
    max_quarter_durations = max_quarter_durations.rename(columns = {'Period_Duration':'Period_Duration_Max'})
    max_quarter_durations = max_quarter_durations.pivot(index = 'Match_ID', columns='Period_Number', values='Period_Duration_Max')
    chains = chains.merge(max_quarter_durations, how='left', on = ['Match_ID'])
    time_seconds = np.where(chains['Period_Number'] == 1, chains['Period_Duration'],
                                np.where(chains['Period_Number'] == 2, chains[1] + chains['Period_Duration'],
                                        np.where(chains['Period_Number'] == 3, chains[1] + chains[2] + chains['Period_Duration'],
                                                    np.where(chains['Period_Number'] == 4, chains[1] + chains[2] + chains[3] + chains['Period_Duration'],
                                                            0))))
    
    return time_seconds

def _create_result(actions):
    
    result = ['success']*len(actions)
        
    result = np.where(actions['Disposal'] == 'effective', 'success', result)
    result = np.where(actions['Disposal'] == 'ineffective', 'fail', result)
    result = np.where(actions['Disposal'] == 'clanger', 'fail', result)
    
    shot_goal = (actions['action_type'] == "shot") & (actions['Final_State'] == "goal")
    shot_behind = (actions['action_type'] == "shot") & (actions['Final_State'] == "behind")
    shot_not_goal_behind = (actions['action_type'] == "shot") & ((actions['Final_State'] != "goal") | (actions['Final_State'] != "behind"))
    result = np.where(shot_goal, "goal",
                      np.where(shot_behind, "behind", 
                               np.where(shot_not_goal_behind, "miss", 
                                        result)))

    result = np.where(actions['action_type'] == 'bounce', 'success', result)
    result = np.where(actions['action_type'] == 'error', 'fail', result)
    result = np.where(actions['action_type'] == 'mark_dropped', 'fail', result)
    result = np.where(actions['action_type'] == 'mark_fumbled', 'fail', result)
    result = np.where(actions['action_type'] == 'non_action', 'non_action', result)

    return result

def _create_bodypart(actions):
    
    body_part = np.where(actions['action_type'].isin(['kick', 'kickin', 'shot']), 'foot', 'hand')
    body_part = np.where(actions['action_type'] == 'non_action', 'non_action', body_part)
    
    return body_part

def _create_start_location(chains):
    
    # Create raw pitch x, y locations (current x, y locations try to always go left to right for both teams)
    start_x = np.where((chains['Home_Team_Direction_Q1'] == "right") & (chains['Team_Chain'] == chains['Away_Team']), 
                                -1*chains['x'],
                                np.where((chains['Home_Team_Direction_Q1'] == "left") & (chains['Team_Chain'] == chains['Home_Team']), 
                                        -1*chains['x'], 
                                        chains['x']))
    start_y = np.where((chains['Home_Team_Direction_Q1'] == "right") & (chains['Team_Chain'] == chains['Away_Team']), 
                                -1*chains['y'],
                                np.where((chains['Home_Team_Direction_Q1'] == "left") & (chains['Team_Chain'] == chains['Home_Team']), 
                                        -1*chains['y'], 
                                        chains['y']))
    
    return start_x, start_y

def _create_end_location(chains):
        
    end_x = np.where(chains['action_type'].isin(['kick', 'handball']), chains['start_x'].shift(-1), chains['start_x'])
    end_y = np.where(chains['action_type'].isin(['kick', 'handball']), chains['start_y'].shift(-1), chains['start_y'])
    
    end_x = np.where(np.isnan(end_x), chains['start_x'], end_x)
    end_y = np.where(np.isnan(end_y), chains['start_y'], end_y)
    
    return end_x, end_y

def _remove_duplicate_actions(actions):

    actions = actions[~((actions['start_x'] == -1*actions['end_x']) & (actions['start_y'] == -1*actions['end_y']) & (actions['Team_Chain'] != actions['Team']))]

    return actions

def _filter_arpadl_columns(actions):
    
    return actions[list(ARPADLSchema.to_schema().columns.keys())]

min_carry_length: float = 3
min_carry_time: float = 2

def _add_carries(actions):
                
    next_actions = actions.shift(-1, fill_value=0)
    same_team = actions['team'] == next_actions['team']
    same_period = actions['period_id'] == next_actions['period_id']

    dx = actions['end_x'] - next_actions['start_x']
    dy = actions['end_y'] - next_actions['start_y']
    far_enough = dx**2 + dy**2 >= min_carry_length**2
    dt = next_actions['time_seconds'] - actions['time_seconds']
    long_enough = dt >= min_carry_time

    curr_gather = (actions['action_type'] == 'gather')
    curr_gather_from_hitout = (actions['action_type'] == 'gather_from_hitout')
    curr_gather_from_opposition = (actions['action_type'] == 'gather_from_opposition')
    curr_gather = (actions['action_type'] == 'gather')
    curr_hard_ball_get = (actions['action_type'] == 'hard_ball_get')
    curr_loose_ball_get = (actions['action_type'] == 'loose_ball_get')
    curr_kickin_playon = (actions['action_type'] == 'kickin_play_on')

    carry_idx = (
        same_team
        & same_period
        & far_enough
        & long_enough
        & (
            curr_gather | 
            curr_gather_from_hitout | 
            curr_gather_from_opposition |
            curr_loose_ball_get | 
            curr_hard_ball_get | 
            curr_kickin_playon)
    )

    carries = pd.DataFrame()
    prev = actions[carry_idx]
    nex = next_actions[carry_idx]
    carries["match_id"] = nex['match_id']
    carries["period_id"] = nex['period_id']
    carries["time_seconds"] = (prev["time_seconds"] + nex["time_seconds"]) / 2

    carries["team"] = nex['team']
    carries["player"] = nex['player']
    carries["start_x"] = prev['end_x']
    carries["start_y"] = prev['end_y']
    carries["end_x"] = nex['start_x']
    carries["end_y"] = nex['start_y']
    carries["bodypart"] = 'hand'
    carries["action_type"] = 'carry'
    carries["result"] = 'success'

    actions = pd.concat([actions, carries], ignore_index=True, sort=False)
    actions = actions.sort_values(["match_id", "period_id", 'time_seconds']).reset_index(drop=True)
                
    return actions
