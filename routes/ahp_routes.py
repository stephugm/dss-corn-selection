"""
Routes for handling AHP method steps.
"""
from flask import request, redirect, url_for, session, flash, jsonify
from . import wizard_bp
import numpy as np
from methods.ahp import calculate_weights, calculate_consistency_ratio, check_alternative_consistency, check_criteria_consistency, calculate_ahp_results
from utils.file_utils import CSVHandler
import json

# Initialize CSV handler
csv_handler = CSVHandler()

def load_pairwise_comparisons(project_id):
    """Load pairwise comparisons from CSV file."""
    comparisons = {}
    filename = 'pairwise_comparisons.csv'
    
    rows = csv_handler.read_all(filename)
    if not rows or len(rows) < 2:  # Check if file is empty or only has header
        return comparisons
        
    fieldnames = rows[0]
    for row in rows[1:]:
        if not row:  # Skip empty rows
            continue
        row_dict = dict(zip(fieldnames, row))
        if row_dict.get('project_id') == project_id:
            i = int(row_dict.get('criterion1', 0))
            j = int(row_dict.get('criterion2', 0))
            comparisons[(i, j)] = float(row_dict.get('value', 1.0))
    
    return comparisons

@wizard_bp.route('/save-criteria-ahp', methods=['POST'])
def save_criteria_ahp():
    """Save criteria and their types for AHP method."""
    project_id = session['project_id']
    criteria = request.form.getlist('criteria[]')
    criteria_types = request.form.getlist('criteria_type[]')
    
    # Read existing criteria
    filename = 'criteria.csv'
    rows = csv_handler.read_all(filename)
    
    # Filter out existing criteria for this project
    existing_criteria = []
    fieldnames = ['project_id', 'name', 'type']
    if rows:
        fieldnames = rows[0]  # Preserve original fieldnames
        existing_criteria = [row for row in rows[1:] if not row or row[0] != project_id]
    
    # Add new criteria
    new_rows = []
    for criterion, c_type in zip(criteria, criteria_types):
        new_rows.append([project_id, criterion, c_type])
    
    # Combine and save all criteria
    all_rows = [fieldnames] + existing_criteria + new_rows
    csv_handler.write_all(filename, all_rows)
    
    # Store in session
    session['criteria'] = criteria
    session['criterion_types'] = criteria_types
    
    return redirect(url_for('wizard.project_wizard', step=3, method='ahp'))
def save_pairwise_comparisons(project_id, criteria, matrix):
    """Save pairwise comparison matrix to CSV."""
    filename = 'pairwise_comparisons.csv'
    
    # Read existing comparisons
    rows = csv_handler.read_all(filename)
    fieldnames = ['project_id', 'criterion1', 'criterion2', 'value']
    
    # Filter out existing comparisons for this project
    existing_rows = []
    if rows:
        fieldnames = rows[0]  # Preserve original fieldnames
        existing_rows = [row for row in rows[1:] if not row or row[0] != project_id]
    
    # Add new comparisons
    new_rows = []
    for i, row in enumerate(matrix):
        for j, value in enumerate(row):
            new_rows.append([project_id, str(i), str(j), str(value)])
    
    # Combine and save all comparisons
    all_rows = [fieldnames] + existing_rows + new_rows
    csv_handler.write_all(filename, all_rows)

@wizard_bp.route('/save-pairwise-comparison', methods=['POST'])
def save_pairwise_comparison():
    """Save pairwise comparison matrix and calculate weights."""
    project_id = session['project_id']
    criteria = session['criteria']
    n = len(criteria)
    
    # Build comparison matrix
    matrix = np.ones((n, n))
    for i in range(n):
        for j in range(i+1, n):
            value = float(request.form[f'comparison_{i}_{j}'])
            matrix[i][j] = value
            matrix[j][i] = 1/value
    
    # Save the raw comparison data
    save_pairwise_comparisons(project_id, criteria, matrix)
    
    # Calculate weights and check consistency
    weights = calculate_weights(matrix)
    cr = calculate_consistency_ratio(matrix, weights)
    
    if cr > 0.1:
        session['error'] = 'Pairwise comparisons are inconsistent (CR > 0.1). Please revise.'
        return redirect(url_for('wizard.project_wizard', step=3, method='ahp'))
    
    # Save weights using CSVHandler
    filename = 'weights.csv'
    rows = csv_handler.read_all(filename)
    fieldnames = ['project_id', 'criterion', 'weight']
    
    # Filter out existing weights for this project
    existing_weights = []
    if rows:
        fieldnames = rows[0]  # Preserve original fieldnames
        existing_weights = [row for row in rows[1:] if not row or row[0] != project_id]
    
    # Add new weights
    new_weights = []
    for criterion, weight in zip(criteria, weights):
        new_weights.append([project_id, criterion, str(float(weight))])
    
    # Combine and save all weights
    all_weights = [fieldnames] + existing_weights + new_weights
    csv_handler.write_all(filename, all_weights)
    
    # Store weights in session
    session['weights'] = weights.tolist()    
    
    return redirect(url_for('wizard.project_wizard', step=4, method='ahp'))

def load_alternative_comparisons(project_id):
    """Load all alternative comparisons for a project."""
    comparisons = {}
    filename = 'alternative_comparisons.csv'
    
    rows = csv_handler.read_all(filename)
    if not rows or len(rows) < 2:  # Check if file is empty or only has header
        return comparisons
    
    fieldnames = rows[0]
    for row in rows[1:]:
        if not row:  # Skip empty rows
            continue
        row_dict = dict(zip(fieldnames, row))
        if row_dict.get('project_id') == project_id:
            criterion = row_dict.get('criterion')
            i = int(row_dict.get('alternative1', 0))
            j = int(row_dict.get('alternative2', 0))
            value = float(row_dict.get('value', 1.0))
            
            if criterion not in comparisons:
                comparisons[criterion] = {}
            comparisons[criterion][(i, j)] = value
    
    return comparisons

def save_alternative_comparison(project_id, criterion, i, j, value):
    """Save a single alternative comparison to CSV."""
    filename = 'alternative_comparisons.csv'
    
    # Read existing comparisons
    rows = csv_handler.read_all(filename)
    fieldnames = ['project_id', 'criterion', 'alternative1', 'alternative2', 'value']
    
    # Filter out existing comparison for this project, criterion and alternatives
    existing_rows = []
    if rows:
        fieldnames = rows[0]  # Preserve original fieldnames
        existing_rows = [
            row for row in rows[1:] 
            if not row or 
               row[0] != project_id or 
               row[1] != criterion or 
               int(row[2]) != i or 
               int(row[3]) != j
        ]
    
    # Add new comparison
    new_row = [project_id, criterion, str(i), str(j), str(value)]
    all_rows = [fieldnames] + existing_rows + [new_row]
    
    # Save all comparisons
    csv_handler.write_all(filename, all_rows)

@wizard_bp.route('/check-alternative-consistency', methods=['POST'])
def check_consistency():
    """Check consistency of alternative comparisons for a criterion.
    
    Returns:
        JSON response with consistency status and ratio.
    """
    if not (project_id := session.get('project_id')):
        return jsonify(error='No project ID found'), 400
    
    data = request.get_json()
    if not (criterion := data.get('criterion')) or not (comparisons := data.get('comparisons', {})):
        return jsonify(error='Missing required parameters'), 400
    
    alt_comparisons = {
        (criterion, int(i), int(j)): float(value)
        for key, value in comparisons.items()
        if len(parts := key.split('_')) == 2
        for i, j in [parts]
    }
    
    if not alt_comparisons:
        return jsonify(error='No valid comparisons found'), 400
    
    alternatives = list(range(len(session.get('alternatives', []))))
    is_consistent, cr = check_alternative_consistency(project_id, criterion, alternatives, alt_comparisons)
    
    return jsonify({
        'is_consistent': bool(is_consistent), 
        'consistency_ratio': float(cr) if cr is not None else 0.0
    })


@wizard_bp.route('/check-criteria-consistency', methods=['POST'])
def check_criteria_consistency_route():
    """Check consistency of criteria pairwise comparisons.
    
    Returns:
        JSON response with consistency status and ratio.
    """
    if not session.get('project_id'):
        return jsonify(error='No project ID found'), 400
    
    if not (comparisons := request.get_json().get('comparisons', {})):
        return jsonify(error='No comparisons provided'), 400
    
    criteria = session.get('criteria', [])
    is_consistent, cr = check_criteria_consistency(comparisons, len(criteria))
    
    return jsonify({
        'is_consistent': bool(is_consistent), 
        'consistency_ratio': float(cr) if cr is not None else 0.0
    })

def save_alternative_weights(project_id, criteria, alternatives, alternative_matrices):
    """Save alternative weights per criterion to a CSV file.
    
    Args:
        project_id (str): ID of the project
        criteria (list): List of criteria names
        alternatives (list): List of alternative names
        alternative_matrices (dict): Dictionary of alternative comparison matrices by criterion
    """
    filename = 'scores.csv'
    fieldnames = ['project_id', 'alternative', 'criterion', 'score']
    
    # Initialize new rows with header
    new_rows = [fieldnames]
    
    # Calculate weights for each alternative per criterion
    for criterion in criteria:
        if criterion not in alternative_matrices:
            continue
            
        matrix = alternative_matrices[criterion]
        # Calculate weights for this criterion's alternative comparisons
        weights = calculate_weights(matrix)
        
        # Add scores for each alternative
        for alt_name, weight in zip(alternatives, weights):
            new_rows.append([
                project_id,
                alt_name,
                criterion,
                str(weight)
            ])
    
    # Read existing scores to preserve other projects' data
    existing_rows = csv_handler.read_all(filename)
    
    # If file exists and has data, filter out current project's data
    if existing_rows and len(existing_rows) > 1:  # Has header and at least one data row
        # Keep header and rows from other projects
        filtered_rows = [existing_rows[0]]  # Keep header
        filtered_rows.extend([row for row in existing_rows[1:] 
                            if not row or row[0] != project_id])
        # Add new rows for current project
        filtered_rows.extend(new_rows[1:])  # Skip header from new_rows
        # Write back to file
        csv_handler.write_all(filename, filtered_rows)
    else:
        # No existing data or empty file, just write the new data
        csv_handler.write_all(filename, new_rows)

@wizard_bp.route('/save-alternative-comparison', methods=['POST'])
def handle_alternative_comparison():
    """Handle saving alternative comparisons from the form."""
    project_id = session.get('project_id')
    if not project_id:
        return redirect(url_for('wizard.project_wizard', step=1, method='ahp'))
    
    # Process each comparison in the form
    for key, value in request.form.items():
        if key.startswith('comparison_'):
            # Extract criterion, i, j from the key (format: comparison_<criterion>_<i>_<j>)
            parts = key.split('_')
            if len(parts) == 4:  # Should be ['comparison', criterion, i, j]
                _, criterion, i, j = parts
                save_alternative_comparison(project_id, criterion, i, j, value)
    
    # Store in session for immediate feedback
    session['alternative_comparisons'] = request.form.to_dict()
    
    
    criteria = session.get('criteria', [])
    alternative_matrices = {}
    for criterion in criteria:
        comparisons = {}
        for i in range(len(session.get('alternatives', []))):
            for j in range(i+1, len(session.get('alternatives', []))):
                key = f'comparison_{criterion}_{i}_{j}'
                if key in request.form:
                    value = float(request.form[key])
                    comparisons[(i, j)] = value
                    comparisons[(j, i)] = 1.0 / value
        
        # Build comparison matrix for this criterion
        n = len(session.get('alternatives', []))
        matrix = np.ones((n, n))
        for (i, j), value in comparisons.items():
            matrix[i][j] = value
            matrix[j][i] = 1.0 / value
        
        alternative_matrices[criterion] = matrix
    
    # Save the weights
    save_alternative_weights(project_id, criteria, session.get('alternatives', []), alternative_matrices)
    # Check consistency for each criterion
    inconsistencies = []
    
    for criterion in criteria:
        # Get all comparisons for this criterion
        criterion_comparisons = {}
        for key, value in request.form.items():
            if key.startswith(f'comparison_{criterion}_'):
                parts = key.split('_')
                if len(parts) == 4:  # format: comparison_criterion_i_j
                    i, j = int(parts[2]), int(parts[3])
                    criterion_comparisons[(i, j)] = float(value)
        
        # Check consistency
        if criterion_comparisons:
            alternatives = list(range(len(session.get('alternatives', []))))
            is_consistent, cr = check_alternative_consistency(
                project_id,
                criterion,
                alternatives,
                { (criterion, i, j): val for (i, j), val in criterion_comparisons.items() }
            )
            if not is_consistent:
                inconsistencies.append({
                    'criterion': criterion,
                    'consistency_ratio': cr
                })
    
    if inconsistencies:
        # Store inconsistencies in session to display to user
        session['comparison_inconsistencies'] = inconsistencies
        flash('Some comparisons have consistency issues. Please review your comparisons.', 'warning')
    
    # Redirect to results or next step
    return redirect(url_for('results', project_id=project_id))
