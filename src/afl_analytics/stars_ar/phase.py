import numpy as np
import pandas as pd

def create_phase(match_actions: pd.DataFrame) -> pd.Series:
    """
    Creates phases based on the given match actions.

    Parameters:
    - match_actions (DataFrame): A DataFrame containing the match actions.

    Returns:
    - phases (Series): A Series containing the phases for each match action.

    """
    
    change_team = match_actions['team'].ne(match_actions['team'].shift())
    prev_shot = match_actions['action_type'].eq("shot").shift()
    
    match_actions['change_team_shot'] = (change_team | prev_shot)
    latest_phase_start_time = 0
    for i, row in match_actions.iterrows():
        if row['change_team_shot']:
            latest_phase_start_time = row['time_seconds']
        match_actions.loc[i, 'phase_time'] = np.where(row['change_team_shot'], 0, row['time_seconds'] - latest_phase_start_time)
        if match_actions.loc[i, 'phase_time'] > 10:
            latest_phase_start_time = row['time_seconds']
        
    prev_mark = match_actions['action_type'].str.contains('mark').shift()
    shot = match_actions['action_type'].eq('shot')

    too_long = (match_actions['phase_time'] >= 10) & ~(shot & prev_mark)
    
    match_actions = match_actions.drop(columns=['change_team_shot', 'phase_time'])
    
    return (change_team | prev_shot | too_long).cumsum()

def create_phases(actions: pd.DataFrame) -> pd.DataFrame:
    """
    Create phases based on the given actions across multiple matches.

    Parameters:
    actions (DataFrame): A DataFrame containing the actions data.

    Returns:
    DataFrame: A DataFrame with an additional 'phase' column indicating the phase for each action.
    """

    actions = actions.sort_values(['match_id', 'time_seconds'])
    actions['phase'] = actions.groupby('match_id').apply(create_phase).reset_index(drop=True)

    return actions

def create_match_id_phase(actions: pd.DataFrame) -> pd.Series:
    """
    Concatenates the 'match_id' and 'phase' columns of the given DataFrame to create a new Series.
    
    Parameters:
        actions (pd.DataFrame): The DataFrame containing the 'match_id' and 'phase' columns.
        
    Returns:
        pd.Series: A new Series obtained by concatenating the 'match_id' and 'phase' columns.
    """
    return actions['match_id'] + '_' + actions['phase'].astype(str)