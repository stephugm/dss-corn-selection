"""
Analytic Hierarchy Process (AHP) implementation.
"""
import numpy as np
import csv
import os

def calculate_weights(pairwise_matrix):
    """Calculate criteria weights using AHP method.
    
    Args:
        pairwise_matrix (np.array): Square matrix of pairwise comparisons
        
    Returns:
        np.array: Normalized weights for each criterion
    """
    # Calculate eigenvalues and eigenvectors
    eigenvals, eigenvecs = np.linalg.eig(pairwise_matrix)
    
    # Find the maximum eigenvalue index
    max_idx = np.argmax(eigenvals)
    
    # Get the eigenvector corresponding to the largest eigenvalue
    weights = np.real(eigenvecs[:, max_idx])
    
    # Normalize the weights
    weights = weights / np.sum(weights)
    
    return np.round(weights, 4)

def calculate_consistency_ratio(pairwise_matrix, weights):
    """Calculate the consistency ratio of the pairwise comparison matrix.
    
    Args:
        pairwise_matrix (np.array): Square matrix of pairwise comparisons
        weights (np.array): Calculated weights
        
    Returns:
        float: Consistency ratio (CR). CR < 0.1 is considered acceptable.
    """
    n = len(pairwise_matrix)
    
    # Calculate lambda max
    lambda_max = np.sum(np.dot(pairwise_matrix, weights) / weights) / n
    
    # Calculate consistency index
    ci = (lambda_max - n) / (n - 1)
    
    # Random consistency index values
    ri_values = {
        1: 0, 2: 0, 3: 0.58, 4: 0.9, 5: 1.12,
        6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49
    }
    
    # Ensure n is in ri_values
    if n not in ri_values:
        raise ValueError(f"Random consistency index (RI) not defined for matrix size {n}.")

    # Calculate consistency ratio
    if ri_values[n] == 0:
        return 0  # Perfect consistency for n <= 2

    cr = ci / ri_values[n]

    # Handle invalid CR values
    if not np.isfinite(cr):
        cr = float('inf')  # Assign infinity if CR is invalid

    return cr

def load_alternative_comparisons(project_id, filename='data/alternative_comparisons.csv'):
    """Load alternative comparisons from CSV.
    
    Args:
        project_id (str): ID of the project
        filename (str): Path to the comparisons CSV file
        
    Returns:
        dict: Dictionary of comparisons with keys (criterion, alt1, alt2)
    """
    comparisons = {}
    
    if not os.path.exists(filename):
        return comparisons
        
    with open(filename, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] == project_id:
                key = (row['criterion'], int(row['alternative1']), int(row['alternative2']))
                comparisons[key] = float(row['value'])
    
    return comparisons

def build_ahp_matrices(project_id, criteria, alternatives, weights_file='data/weights.csv'):
    """Build pairwise comparison matrices for AHP.
    
    Args:
        project_id (str): ID of the project
        criteria (list): List of criteria
        alternatives (list): List of alternatives
        weights_file (str): Path to the weights CSV file
        
    Returns:
        tuple: (decision_matrix, criteria_weights)
    """
    # Load criteria weights
    criteria_weights = {}
    with open(weights_file, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] == project_id:
                criteria_weights[row['criterion']] = float(row['weight'])
    
    # Load alternative comparisons
    comparisons = load_alternative_comparisons(project_id)
    
    # Build alternative comparison matrices for each criterion
    n_alternatives = len(alternatives)
    n_criteria = len(criteria)
    
    # Initialize matrices
    alternative_matrices = {}
    
    for c, criterion in enumerate(criteria):
        # Create n x n matrix for this criterion
        matrix = np.ones((n_alternatives, n_alternatives))
        
        # Fill in the upper triangle
        for i in range(n_alternatives):
            for j in range(i+1, n_alternatives):
                # Try both (c,i,j) and (c,j,i) order
                if (criterion, i, j) in comparisons:
                    value = comparisons[(criterion, i, j)]
                    matrix[i, j] = value
                    matrix[j, i] = 1.0 / value
                elif (criterion, j, i) in comparisons:
                    value = comparisons[(criterion, j, i)]
                    matrix[i, j] = 1.0 / value
                    matrix[j, i] = value
        
        alternative_matrices[criterion] = matrix
    
    # Calculate priority vectors for each criterion
    priority_vectors = {}
    for criterion in criteria:
        matrix = alternative_matrices[criterion]
        # Calculate priority vector using geometric mean method
        geom_means = np.prod(matrix, axis=1) ** (1.0 / n_alternatives)
        priority_vectors[criterion] = geom_means / np.sum(geom_means)
    
    # Build final decision matrix
    decision_matrix = np.zeros((n_alternatives, n_criteria))
    for j, criterion in enumerate(criteria):
        decision_matrix[:, j] = priority_vectors[criterion]
    
    # Get criteria weights as array
    criteria_weights_array = np.array([criteria_weights.get(c, 0) for c in criteria])
    
    return decision_matrix, criteria_weights_array

def calculate_scores(alternatives_matrix, criteria_weights):
    """Calculate final scores for alternatives using AHP.
    
    Args:
        alternatives_matrix (np.array): Matrix of alternative scores for each criterion
        criteria_weights (np.array): Weights of criteria
        
    Returns:
        np.array: Final scores for each alternative
    """
    # Simple weighted sum
    return np.dot(alternatives_matrix, criteria_weights)

def check_criteria_consistency(comparisons, n_criteria):
    """Check if the criteria comparisons are consistent.
    
    Args:
        comparisons (dict): Dictionary of comparison values with keys 'i_j' and float values
        n_criteria (int): Number of criteria
        
    Returns:
        tuple: (is_consistent, cr) where is_consistent is a boolean and cr is the consistency ratio
    """
    if n_criteria < 2:
        return True, 0.0  # No inconsistency with 0 or 1 criterion
    
    # Build comparison matrix
    matrix = np.ones((n_criteria, n_criteria))
    
    for key, value in comparisons.items():
        try:
            i, j = map(int, key.split('_'))
            if 0 <= i < n_criteria and 0 <= j < n_criteria:
                matrix[i][j] = float(value)
                matrix[j][i] = 1 / float(value)
        except (ValueError, IndexError):
            continue
    
    # Calculate weights and consistency ratio
    weights = calculate_weights(matrix)
    cr = calculate_consistency_ratio(matrix, weights)
  
    return cr <= 0.1, cr

def check_alternative_consistency(project_id, criterion, alternatives, comparisons):
    """Check if the alternative comparisons for a criterion are consistent.
    
    Args:
        project_id (str): ID of the project
        criterion (str): The criterion being evaluated
        alternatives (list): List of alternative indices
        comparisons (dict): Dictionary of comparison values with keys (criterion, i, j)
        
    Returns:
        tuple: (is_consistent, cr) where is_consistent is a boolean and cr is the consistency ratio
    """
    n = len(alternatives)
    if n < 2:
        return True, 0.0  # No inconsistency with 0 or 1 alternative
    # Build comparison matrix
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            key1 = (criterion, alternatives[i], alternatives[j])
            key2 = (criterion, alternatives[j], alternatives[i])
            
            if key1 in comparisons:
                value = comparisons[key1]
                matrix[i][j] = value
                matrix[j][i] = 1 / value
            elif key2 in comparisons:
                value = comparisons[key2]
                matrix[j][i] = value
                matrix[i][j] = 1 / value
    # Calculate weights and consistency ratio
    weights = calculate_weights(matrix)
    cr = calculate_consistency_ratio(matrix, weights)
    return cr <= 0.1, cr

def calculate_ahp_results(project_id, criteria, alternatives):
    """Calculate AHP results for a project.
    
    Args:
        project_id (str): ID of the project
        criteria (list): List of criteria
        alternatives (list): List of alternatives
        
    Returns:
        np.array: Final scores for each alternative
    """
    # Build AHP matrices
    decision_matrix, criteria_weights = build_ahp_matrices(project_id, criteria, alternatives)
    
    # Calculate final scores
    final_scores = calculate_scores(decision_matrix, criteria_weights)
    
    return final_scores
