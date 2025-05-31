"""
Decision Support System methods package.
"""
from . import ahp, topsis, profile_matching

def calculate_scores(method, **kwargs):
    """Calculate scores using the specified method.
    
    Args:
        method (str): Name of the method to use ('ahp', 'topsis', or 'profile_matching')
        **kwargs: Method-specific arguments
        
    Returns:
        np.array: Final scores for each alternative
    """
    method_map = {
        'ahp': ahp.calculate_scores,
        'topsis': topsis.calculate_scores,
        'profile_matching': profile_matching.calculate_scores
    }
    
    if method not in method_map:
        raise ValueError(f"Unknown method: {method}")
    
    return method_map[method](**kwargs)
