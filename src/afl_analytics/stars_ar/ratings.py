import numpy as np
import pandas as pd
import random
from dtaidistance import dtw_ndim
from afl_analytics.stars_ar.phase import create_phases, create_match_id_phase
from afl_analytics.stars_ar.clustering import hierarchical_clustering

def get_phases(actions: pd.DataFrame) -> list[np.array]:
    """
    Retrieves the phases from the given actions DataFrame.

    Parameters:
    actions (pd.DataFrame): The DataFrame containing the actions data.

    Returns:
    list[np.array]: A list of NumPy arrays, where each array represents the start and end coordinates of actions in a specific phase.
    """
    return [np.array(actions[actions['match_id_phase'] == x][['start_x', 'start_y', 'end_x', 'end_y']]) for x in list(actions['match_id_phase'].unique())]

def do_phase_clustering(phases: list[np.array], n_clusters: int = 20) -> list[int]:
    """
    Perform phase clustering on a list of phases using dynamic time warping (DTW).

    Args:
        phases (list[np.array]): A list of numpy arrays representing phases.
        n_clusters (int, optional): The number of clusters to create. Defaults to 20.

    Returns:
        list[int]: A list of cluster labels assigned to each phase.

    """
    distances = dtw_ndim.distance_matrix(phases)
    labels = hierarchical_clustering(distances, n_clusters=n_clusters)
    
    return labels

def create_phase_score(actions: pd.DataFrame) -> pd.Series:
    """
    Calculates the phase score for each action in the given DataFrame.
    
    Parameters:
        actions (pd.DataFrame): A DataFrame containing the actions data.
        
    Returns:
        pd.Series: A Series containing the phase scores for each action.
    """
    phase_result = actions.groupby('match_id_phase')['result'].last()
    phase_result_score = ((phase_result == 'goal') | (phase_result == "behind")).reset_index()
    phase_result_score_map = dict(zip(phase_result_score['match_id_phase'], phase_result_score['result']))
    
    return actions['match_id_phase'].map(phase_result_score_map)*1

def create_phase_ratings(actions: pd.DataFrame) -> pd.Series:
    """
    Create phase ratings based on the average phase score for each label.

    Parameters:
    actions (pd.DataFrame): A DataFrame containing the actions data.

    Returns:
    pd.Series: A Series containing the phase ratings for each action label.
    """
    phase_ratings = actions.groupby('label').mean()['phase_score']
    phase_ratings_map = dict(phase_ratings)
    
    return actions['label'].map(phase_ratings_map)

def exponential_decay(length: int) -> np.array:
    """
    Calculate the exponential decay values for a given length.

    Parameters:
    length (int): The length of the decay array.

    Returns:
    np.array: An array of exponential decay values.

    Example:
    >>> exponential_decay(5)
    array([1.        , 2.71828183, 7.3890561 , 20.08553692, 54.59815003])
    """
    decay = np.exp(np.arange(length))
    return decay / np.sum(decay)

def group_exponential_decay(group: pd.DataFrame) -> np.array:
    """
    Applies exponential decay to a group of data.

    Parameters:
        group (pd.groupby): The group of data to apply exponential decay to.

    Returns:
        np.array: The result of applying exponential decay to the group of data.
    """
    return exponential_decay(len(group))

def create_exponential_decay_weights(actions: pd.DataFrame) -> np.array:
    """
    Create exponential decay weights based on the given actions.

    Parameters:
        actions (pd.DataFrame): A DataFrame containing the actions data.

    Returns:
        np.array: An array of exponential decay weights.

    """
    return np.concatenate(actions.groupby('match_id_phase').apply(group_exponential_decay)).ravel()

def create_action_rating(actions: pd.DataFrame) -> pd.Series:
    """
    Calculates the action rating by multiplying the phase rating with the weights.
    
    Parameters:
        actions (pd.DataFrame): A DataFrame containing the actions data.
        
    Returns:
        pd.Series: A Series containing the action ratings.
    """
    return actions['phase_rating'] * actions['weights']

# def action_rating(actions, sample: int = 5000):
    
#     sample_match_id_phase = random.sample(list(actions['match_id_phase'].unique()), sample)
#     sample_match_id_phase.sort()

#     sample_actions = actions[actions['match_id_phase'].isin(sample_match_id_phase)]
#     sample_phases = get_phases(sample_actions)
    
#     labels = do_phase_clustering(sample_phases, n_clusters=20)
    
#     match_id_phase_label_map = dict(zip(sample_match_id_phase, labels))
#     sample_actions['label'] = sample_actions['match_id_phase'].map(match_id_phase_label_map)
    
#     sample_actions['phase_score'] = create_phase_score(sample_actions)
#     sample_actions['phase_rating'] = create_phase_ratings(sample_actions)
#     sample_actions['weights'] = create_exponential_decay_weights(sample_actions)
    
#     return create_action_rating(sample_actions)