"""
Technique for Order Preference by Similarity to Ideal Solution (TOPSIS) implementation.
"""
import numpy as np

def normalize_decision_matrix(matrix):
    """Normalize the decision matrix using vector normalization.
    
    Args:
        matrix (np.array): Decision matrix with alternatives as rows and criteria as columns
        
    Returns:
        np.array: Normalized decision matrix
    """
    return matrix / np.sqrt(np.sum(matrix ** 2, axis=0))

def calculate_weighted_normalized_matrix(normalized_matrix, weights):
    """Calculate the weighted normalized decision matrix.
    
    Args:
        normalized_matrix (np.array): Normalized decision matrix
        weights (np.array): Criteria weights
        
    Returns:
        np.array: Weighted normalized decision matrix
    """
    return normalized_matrix * weights

def find_ideal_solutions(weighted_matrix, criteria_types):
    """Find positive and negative ideal solutions.
    
    Args:
        weighted_matrix (np.array): Weighted normalized decision matrix
        criteria_types (list): List of criteria types ('benefit' or 'cost')
        
    Returns:
        tuple: (positive ideal solution, negative ideal solution)
    """
    positive_ideal = []
    negative_ideal = []
    
    for i, criterion_type in enumerate(criteria_types):
        if criterion_type == 'benefit':
            positive_ideal.append(np.max(weighted_matrix[:, i]))
            negative_ideal.append(np.min(weighted_matrix[:, i]))
        else:  # cost criterion
            positive_ideal.append(np.min(weighted_matrix[:, i]))
            negative_ideal.append(np.max(weighted_matrix[:, i]))
    
    return np.array(positive_ideal), np.array(negative_ideal)

def calculate_separation_measures(weighted_matrix, positive_ideal, negative_ideal):
    """Calculate separation measures from positive and negative ideal solutions.
    
    Args:
        weighted_matrix (np.array): Weighted normalized decision matrix
        positive_ideal (np.array): Positive ideal solution
        negative_ideal (np.array): Negative ideal solution
        
    Returns:
        tuple: (separation from positive ideal, separation from negative ideal)
    """
    s_positive = np.sqrt(np.sum((weighted_matrix - positive_ideal) ** 2, axis=1))
    s_negative = np.sqrt(np.sum((weighted_matrix - negative_ideal) ** 2, axis=1))
    
    return s_positive, s_negative

def calculate_relative_closeness(s_positive, s_negative):
    """Calculate relative closeness to the ideal solution.
    
    Args:
        s_positive (np.array): Separation from positive ideal solution
        s_negative (np.array): Separation from negative ideal solution
        
    Returns:
        np.array: Relative closeness coefficients
    """
    return s_negative / (s_positive + s_negative)

def calculate_scores(decision_matrix, weights, criteria_types):
    """Calculate final scores using TOPSIS method.
    
    Args:
        decision_matrix (np.array): Matrix of alternative scores for each criterion
        weights (np.array): Criteria weights
        criteria_types (list): List indicating if each criterion is 'benefit' or 'cost'
        
    Returns:
        np.array: Final scores for each alternative
    """
    # Normalize the decision matrix
    normalized_matrix = normalize_decision_matrix(decision_matrix)
    
    # Calculate weighted normalized matrix
    weighted_matrix = calculate_weighted_normalized_matrix(normalized_matrix, weights)
    
    # Find ideal solutions
    positive_ideal, negative_ideal = find_ideal_solutions(weighted_matrix, criteria_types)
    
    # Calculate separation measures
    s_positive, s_negative = calculate_separation_measures(
        weighted_matrix, positive_ideal, negative_ideal
    )
    
    # Calculate relative closeness to ideal solution
    scores = calculate_relative_closeness(s_positive, s_negative)
    
    return scores
