"""
Common routes shared between all methods.
"""
from itertools import pairwise
from flask import render_template, request, redirect, url_for, session
from . import wizard_bp
import csv
from datetime import datetime
import os
from .ahp_routes import load_pairwise_comparisons, load_alternative_comparisons

PROJECTS_FILE = 'data/projects.csv'
CRITERIA_FILE = 'data/criteria.csv'
ALTERNATIVES_FILE = 'data/alternatives.csv'
SCORES_FILE = 'data/scores.csv'

def ensure_data_files():
    """Ensure all required data files exist."""
    os.makedirs('data', exist_ok=True)
    files = [PROJECTS_FILE, CRITERIA_FILE, ALTERNATIVES_FILE, SCORES_FILE]
    for file in files:
        if not os.path.exists(file):
            with open(file, 'w', newline='') as f:
                writer = csv.writer(f)
                if file == PROJECTS_FILE:
                    writer.writerow(['project_id', 'title', 'description', 'method', 'created_at'])
                elif file == CRITERIA_FILE:
                    writer.writerow(['project_id', 'criterion', 'type'])
                elif file == ALTERNATIVES_FILE:
                    writer.writerow(['project_id', 'alternative'])
                elif file == SCORES_FILE:
                    writer.writerow(['project_id', 'alternative', 'criterion', 'score'])

@wizard_bp.route('/project-wizard/<int:step>')
def project_wizard(step):
    """Handle project wizard navigation."""
    method = request.args.get('method') or session.get('method')
    if not method:
        return redirect(url_for('method_selection'))
    
    # Store method in session
    session['method'] = method
    
    template = f'{method}_steps.html'
    total_steps = 5  # All methods have 5 steps
    
    pairwise_scale_options = [
        {'value': 9, 'label': '9'},
        {'value': 7, 'label': '7'},
        {'value': 5, 'label': '5'},
        {'value': 3, 'label': '3'},
        {'value': 2, 'label': '2'},
        {'value': 1, 'label': '1'},
        {'value': 1/2, 'label': '1/2'},
        {'value': 1/3, 'label': '1/3'},
        {'value': 1/5, 'label': '1/5'},
        {'value': 1/7, 'label': '1/7'},
        {'value': 1/9, 'label': '1/9'}
    ]
    
    # Load comparisons based on step and method
    pairwise_comparisons = {}
    alternative_comparisons = {}
    project_id = session.get('project_id')
    if not project_id:
        return redirect(url_for('dashboard'))
    
    if project_id:
        if step == 3 and method == 'ahp':
            pairwise_comparisons = load_pairwise_comparisons(project_id)
        elif step == 5 and method == 'ahp':
            alternative_comparisons = load_alternative_comparisons(project_id)
    return render_template(
        f'wizard/{template}',
        current_step=step,
        total_steps=total_steps,
        pairwise_scale_options=pairwise_scale_options,
        criteria=session.get('criteria', []),
        criterion_types=session.get('criterion_types', []),
        alternatives=session.get('alternatives', []),
        weights=session.get('weights', {}),
        pairwise_comparisons=pairwise_comparisons,
        alternative_comparisons=alternative_comparisons
    )

@wizard_bp.route('/save-project-info', methods=['POST'])
def save_project_info():
    """Save project information and start the wizard."""
    ensure_data_files()

    print(request.form)
    
    title = request.form['title']
    description = request.form.get('description', '')
    method = request.form['method']
    
    # Check if we're in edit mode
    project_id = session.get('project_id')
    if project_id:
        # Update existing project
        projects = []
        with open(PROJECTS_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['project_id'] == project_id:
                    row.update({
                        'title': title,
                        'description': description,
                        'method': method
                    })
                projects.append(row)
        
        # Write back all projects
        with open(PROJECTS_FILE, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['project_id', 'title', 'description', 'method', 'created_at'])
            writer.writeheader()
            writer.writerows(projects)
    else:
        # Generate new project ID and save to CSV
        project_id = datetime.now().strftime('%Y%m%d%H%M%S')
        with open(PROJECTS_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([project_id, title, description, method, datetime.now().isoformat()])
    
    # Store in session
    session['project_id'] = project_id
    session['method'] = method
    
    return redirect(url_for('wizard.project_wizard', step=2, method=method))

@wizard_bp.route('/save-alternatives', methods=['POST'])
def save_alternatives():
    """Save alternatives to CSV and session."""
    project_id = session['project_id']
    alternatives = request.form.getlist('alternatives[]')
    
    # Read existing alternatives
    existing_alternatives = []
    with open(ALTERNATIVES_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] != project_id:
                existing_alternatives.append(row)
    
    # Add new alternatives
    for alt in alternatives:
        existing_alternatives.append({
            'project_id': project_id,
            'alternative': alt
        })
    
    # Write back all alternatives
    with open(ALTERNATIVES_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['project_id', 'alternative'])
        writer.writeheader()
        writer.writerows(existing_alternatives)
    
    # Store in session
    session['alternatives'] = alternatives
    
    return redirect(url_for('wizard.project_wizard', step=5, method=session['method']))

@wizard_bp.route('/save-alternative-scores-topsis', methods=['POST'])
def save_alternative_scores():
    """Save alternative scores to CSV and session."""
    project_id = session['project_id']
    alternatives = session['alternatives']
    criteria = session['criteria']
    scores = {}

    # Delete existing scores for this project
    existing_scores = []
    with open(SCORES_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] != project_id:
                existing_scores.append(row)

    # Get new scores from form
    for alternative in alternatives:
        scores[alternative] = {}
        for criterion in criteria:
            score_key = f"score_{alternative.replace(' ', '_')}_{criterion.replace(' ', '_')}"
            scores[alternative][criterion] = float(request.form[score_key])
            existing_scores.append({
                'project_id': project_id,
                'alternative': alternative,
                'criterion': criterion,
                'score': scores[alternative][criterion]
            })

    # Write all scores back to CSV
    with open(SCORES_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['project_id', 'alternative', 'criterion', 'score'])
        writer.writeheader()
        writer.writerows(existing_scores)

    # Store in session
    session['scores'] = scores

    return redirect(url_for('wizard.project_wizard', step=6, method=session['method']))
