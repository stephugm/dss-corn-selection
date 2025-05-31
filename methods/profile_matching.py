"""
Profile Matching method implementation.
"""
import numpy as np

def calculate_gap(target_profile, actual_profile):
    """Calculate the gap between target profile and actual profile.
    
    Args:
        target_profile (np.array): Target/ideal profile values
        actual_profile (np.array): Actual profile values
        
    Returns:
        np.array: Gap values
    """
    return actual_profile - target_profile

def map_gap_to_weight(gap):
    """Map gap value to weight based on GAP weight table.
    
    Args:
        gap (float or np.array): Gap value(s)
        
    Returns:
        float or np.array: Weight value(s) based on gap
    """
    gap_weights = {
        0: 5,    # No gap
        1: 4.5,  # Individual exceeds by 1
        -1: 4,   # Competency shortage of 1
        2: 3.5,  # Individual exceeds by 2
        -2: 3,   # Competency shortage of 2
        3: 2.5,  # Individual exceeds by 3
        -3: 2,   # Competency shortage of 3
        4: 1.5,  # Individual exceeds by 4
        -4: 1    # Competency shortage of 4
    }
    
    if hasattr(gap, '__iter__') or isinstance(gap, np.ndarray):
        # Handle numpy array or iterable
        gap = np.asarray(gap, dtype=float)
        # Clip values to [-4, 4] range and round to nearest integer
        capped_gap = np.clip(np.rint(gap), -4, 4).astype(int)
        # Vectorized lookup using np.vectorize
        vfunc = np.vectorize(lambda x: gap_weights.get(int(x), 0.0))
        return vfunc(capped_gap)
    else:
        # Handle single value
        capped_gap = int(round(float(gap)))
        capped_gap = max(min(capped_gap, 4), -4)
        return gap_weights.get(capped_gap, 0.0)

def calculate_core_factor_score(weights, core_criteria_indices):
    """Calculate core factor score.
    
    Args:
        weights (np.array): Mapped weights from gap values
        core_criteria_indices (list): Indices of core criteria
        
    Returns:
        float: Core factor score
    """
    core_weights = weights[core_criteria_indices]
    return np.mean(core_weights)

def calculate_secondary_factor_score(weights, secondary_criteria_indices):
    """Calculate secondary factor score.
    
    Args:
        weights (np.array): Mapped weights from gap values
        secondary_criteria_indices (list): Indices of secondary criteria
        
    Returns:
        float: Secondary factor score
    """
    secondary_weights = weights[secondary_criteria_indices]
    return np.mean(secondary_weights)

def calculate_scores(alternatives_matrix, target_profile, core_indices, 
                    secondary_indices, core_weight=0.6, secondary_weight=0.4):
    """Calculate final scores using Profile Matching method.
    
    Args:
        alternatives_matrix (np.array): Matrix of alternative scores for each criterion
        target_profile (np.array): Target/ideal profile values
        core_indices (list): Indices of core criteria
        secondary_indices (list): Indices of secondary criteria
        core_weight (float): Weight for core factors (default: 0.6)
        secondary_weight (float): Weight for secondary factors (default: 0.4)
        
    Returns:
        np.array: Final scores for each alternative
    """
    n_alternatives = len(alternatives_matrix)
    scores = np.zeros(n_alternatives)
    for i in range(n_alternatives):
        # Calculate gaps
        gaps = calculate_gap(target_profile, alternatives_matrix[i])
        
        # Map gaps to weights
        weights = np.array([map_gap_to_weight(gap) for gap in gaps])
        
        # Calculate core and secondary factor scores
        core_score = calculate_core_factor_score(weights, core_indices)
        secondary_score = calculate_secondary_factor_score(weights, secondary_indices)
        
        # Calculate final score
        scores[i] = (core_weight * core_score) + (secondary_weight * secondary_score)
    
    return scores
