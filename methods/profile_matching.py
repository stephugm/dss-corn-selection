"""
Profile Matching method implementation with global min-max scaling per criterion.
"""
import numpy as np

def scale_value(value, global_min_val, global_max_val):
    """
    Scales a single value to a range of 1 to 5 using global min-max normalization.
    If the global min and max are the same (e.g., all values for a criterion are identical),
    it returns 3.0 (middle of the 1-5 scale).

    Args:
        value (float): The single numerical value to scale.
        global_min_val (float): The minimum value observed for this criterion across all data.
        global_max_val (float): The maximum value observed for this criterion across all data.

    Returns:
        float: The scaled value between 1 and 5.
    """
    if global_max_val - global_min_val == 0:
        return 3.0  # Middle of 1-5 scale if all values for this criterion are identical
    # Scale to [1,5] using the global min and max for the criterion
    return 1 + 4 * (value - global_min_val) / (global_max_val - global_min_val)

def compute_global_min_max_per_criterion(alternatives_matrix, target_profile):
    """
    Computes the global minimum and maximum values for each criterion
    across all alternatives and the target profile.

    Args:
        alternatives_matrix (np.array): Matrix of alternative scores for each criterion.
        target_profile (np.array): Target/ideal profile values.

    Returns:
        tuple: A tuple containing two np.arrays:
               - global_mins (np.array): Minimum value for each criterion.
               - global_maxs (np.array): Maximum value for each criterion.
    """
    # Combine target profile and all alternatives into a single dataset for min/max calculation
    all_data = np.vstack((target_profile, alternatives_matrix))
    
    # Calculate min and max for each column (criterion)
    global_mins = np.min(all_data, axis=0)
    global_maxs = np.max(all_data, axis=0)
    
    return global_mins, global_maxs

def calculate_gap(target_profile, actual_profile, global_mins, global_maxs):
    """Calculate the gap between scaled target profile and actual profile
    using globally determined min/max values for scaling each criterion.

    Args:
        target_profile (np.array): Target/ideal profile values.
        actual_profile (np.array): Actual profile values.
        global_mins (np.array): Global minimum value for each criterion.
        global_maxs (np.array): Global maximum value for each criterion.

    Returns:
        np.array: Gap values (scaled_actual - scaled_target).
    """
    n_criteria = len(target_profile)
    scaled_target = np.zeros(n_criteria)
    scaled_actual = np.zeros(n_criteria)

    # Scale each value using its corresponding global min/max for that criterion
    for j in range(n_criteria):
        scaled_target[j] = scale_value(target_profile[j], global_mins[j], global_maxs[j])
        scaled_actual[j] = scale_value(actual_profile[j], global_mins[j], global_maxs[j])
    
    print(f"Scaled Target: {scaled_target}, Scaled Actual: {scaled_actual}")
    print(f"Target Profile: {target_profile}, Actual Profile: {actual_profile}")
    
    return scaled_actual - scaled_target

def map_gap_to_weight(gap):
    """Map gap value to weight based on a predefined GAP weight table.
    
    Args:
        gap (float or np.array): Gap value(s)
        
    Returns:
        float or np.array: Weight value(s) based on gap
    """
    gap_weights = {
        0: 5,    # No gap, perfect match
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
    """Calculate final scores using Profile Matching method with global scaling per criterion.
    
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

    # STEP 1: Compute global min and max for each criterion across all data
    global_mins, global_maxs = compute_global_min_max_per_criterion(alternatives_matrix, target_profile)
    print(f"Global Mins per criterion: {global_mins}")
    print(f"Global Maxs per criterion: {global_maxs}\n")

    for i in range(n_alternatives):
        print(f"--- Processing Alternative {i+1} ---")
        # STEP 2: Calculate gaps using the global min/max for scaling
        gaps = calculate_gap(target_profile, alternatives_matrix[i], global_mins, global_maxs)
        print(f"Calculated Gaps: {gaps}")
        
        # STEP 3: Map gaps to weights
        weights = np.array([map_gap_to_weight(gap) for gap in gaps])
        print(f"Mapped Weights: {weights}")
        
        # STEP 4: Calculate core and secondary factor scores
        core_score = calculate_core_factor_score(weights, core_indices)
        secondary_score = calculate_secondary_factor_score(weights, secondary_indices)
        
        # STEP 5: Calculate final score
        scores[i] = (core_weight * core_score) + (secondary_weight * secondary_score)
        print(f"Core Score: {core_score:.2f}, Secondary Score: {secondary_score:.2f}, Final Score: {scores[i]:.2f}\n")
    
    return scores
