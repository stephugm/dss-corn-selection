"""
Routes for handling TOPSIS method steps.
"""
from flask import request, redirect, url_for, session
from . import wizard_bp
import csv
from methods.topsis import normalize_decision_matrix, calculate_weighted_normalized_matrix, find_ideal_solutions, calculate_scores

@wizard_bp.route('/save-criteria-topsis', methods=['POST'])
def save_criteria_topsis():
    """Save criteria and their types for TOPSIS method."""
    project_id = session['project_id']
    criteria = request.form.getlist('criteria[]')
    criteria_types = request.form.getlist('criteria_type[]')
    
    # Read existing criteria
    existing_criteria = []
    with open('data/criteria.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] != project_id:
                existing_criteria.append(row)
    
    # Add new criteria
    for criterion, c_type in zip(criteria, criteria_types):
        existing_criteria.append({
            'project_id': project_id,
            'name': criterion,
            'type': c_type
        })
    
    # Write back all criteria
    with open('data/criteria.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['project_id', 'name', 'type'])
        writer.writeheader()
        writer.writerows(existing_criteria)
    
    # Store in session
    session['criteria'] = criteria
    session['criterion_types'] = criteria_types
    
    return redirect(url_for('wizard.project_wizard', step=3, method='topsis'))

@wizard_bp.route('/save-weights-topsis', methods=['POST'])
def save_weights_topsis():
    """Save criteria weights for TOPSIS method."""
    project_id = session['project_id']
    criteria = session['criteria']
    weights = request.form.getlist('weights[]')
    
    # Convert weights to float and normalize
    weights = [float(w) for w in weights]
    total = sum(weights)
    weights = [w/total for w in weights]
    
    # Read existing weights
    existing_weights = []
    with open('data/weights.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] != project_id:
                existing_weights.append(row)
    
    # Add new weights
    for criterion, weight in zip(criteria, weights):
        existing_weights.append({
            'project_id': project_id,
            'criterion': criterion,
            'weight': weight
        })
    
    # Write back all weights
    with open('data/weights.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['project_id', 'criterion', 'weight'])
        writer.writeheader()
        writer.writerows(existing_weights)
        
    session_weights = {}
    for criterion, weight in zip(criteria, weights):
        session_weights[criterion] = weight
    # Store in session
    session['weights'] = session_weights
    
    return redirect(url_for('wizard.project_wizard', step=4, method='topsis'))

@wizard_bp.route('/save-alternative-scores-topsis', methods=['POST'])
def save_alternative_scores_topsis():
    """Save alternative scores for TOPSIS."""
    project_id = session['project_id']
    criteria = session['criteria']
    alternatives = session['alternatives']
    
    # Read existing scores
    existing_scores = []
    with open('data/scores.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] != project_id:
                existing_scores.append(row)
    
    # Get scores from form
    scores = {}
    for alt in alternatives:
        scores[alt] = {}
        for crit in criteria:
            score_key = f"score_{alt.replace(' ', '_')}_{crit.replace(' ', '_')}"
            score = float(request.form[score_key])
            
            # Add to existing scores
            existing_scores.append({
                'project_id': project_id,
                'alternative': alt,
                'criterion': crit,
                'score': score
            })
            scores[alt][crit] = score
    
    # Write back all scores
    with open('data/scores.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['project_id', 'alternative', 'criterion', 'score'])
        writer.writeheader()
        writer.writerows(existing_scores)
    
    # Store in session for later use
    session['scores'] = scores
    
    return redirect(url_for('results', project_id=project_id))
