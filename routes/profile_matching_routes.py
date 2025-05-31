"""
Routes for handling Profile Matching method steps.
"""
from flask import request, redirect, url_for, session
from . import wizard_bp
import csv

@wizard_bp.route('/save-criteria-profile', methods=['POST'])
def save_criteria_profile():
    """Save criteria and their types (core/secondary) for Profile Matching method."""
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
    
    return redirect(url_for('wizard.project_wizard', step=3, method='profile_matching'))

@wizard_bp.route('/save-target-profile', methods=['POST'])
def save_target_profile():
    """Save target profile values for Profile Matching method."""
    project_id = session['project_id']
    criteria = session['criteria']
    target_values = request.form.getlist('target_values[]')
    
    # Convert to integers
    target_values = [int(val) for val in target_values]
    
    # Read existing target profiles
    existing_profiles = []
    with open('data/target_profiles.csv', 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] != project_id:
                existing_profiles.append(row)
    
    # Add new target values
    for criterion, value in zip(criteria, target_values):
        existing_profiles.append({
            'project_id': project_id,
            'criterion': criterion,
            'target_value': value
        })
    
    # Write back all target profiles
    with open('data/target_profiles.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['project_id', 'criterion', 'target_value'])
        writer.writeheader()
        writer.writerows(existing_profiles)
    
    # Store in session
    session['target_profile'] = target_values
    
    return redirect(url_for('wizard.project_wizard', step=4, method='profile_matching'))

@wizard_bp.route('/save-alternative-scores-profile', methods=['POST'])
def save_alternative_scores_profile():
    """Save alternative scores for Profile Matching."""
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
