from flask import Flask, render_template, redirect, url_for, session, jsonify, Response, send_file
import numpy as np
import csv
import os
from routes import wizard_bp
from methods import ahp
from utils.file_utils import CSVHandler
import methods.topsis as topsis
import methods.profile_matching as profile_matching
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import io
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)

# Register blueprints
app.register_blueprint(wizard_bp, url_prefix='/wizard')

# Routes
@app.route('/')
def index():
    """Display the dashboard page."""    
    return redirect(url_for('dashboard'))

# Data storage paths
DATA_DIR = 'data'
os.makedirs(DATA_DIR, exist_ok=True)

PROJECTS_FILE = os.path.join(DATA_DIR, 'projects.csv')
CRITERIA_FILE = os.path.join(DATA_DIR, 'criteria.csv')
ALTERNATIVES_FILE = os.path.join(DATA_DIR, 'alternatives.csv')
WEIGHTS_FILE = os.path.join(DATA_DIR, 'weights.csv')
SCORES_FILE = os.path.join(DATA_DIR, 'scores.csv')
RESULTS_FILE = os.path.join(DATA_DIR, 'results.csv')
TARGET_PROFILES_FILE = os.path.join(DATA_DIR, 'target_profiles.csv')

# Initialize CSV files if they don't exist
def init_csv_files():
    print('Initializing CSV files', flush=True)
    files = {
        PROJECTS_FILE: ['project_id', 'title', 'description', 'method', 'created_at'],
        CRITERIA_FILE: ['project_id', 'criterion', 'type'],
        ALTERNATIVES_FILE: ['project_id', 'alternative'],
        WEIGHTS_FILE: ['project_id', 'criterion', 'weight'],
        SCORES_FILE: ['project_id', 'alternative', 'criterion', 'score'],
        RESULTS_FILE: ['project_id', 'alternative', 'final_score'],
        TARGET_PROFILES_FILE: ['project_id', 'criterion', 'target_value']
    }
    
    for file, headers in files.items():
        if not os.path.exists(file):
            with open(file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)

# Data management functions
def get_projects():
    """Get all projects."""
    projects = []
    with open(PROJECTS_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            projects.append(row)
    return projects

def get_project(project_id):
    """Get a specific project by ID."""
    with open(PROJECTS_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] == project_id:
                return row
    return None

def load_weights_from_csv(project_id):
    weights = {}
    with open(WEIGHTS_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] == project_id:
                weights[row['criterion']] = float(row['weight'])
   
    return weights

def recalculate_pairwise(weights):
    """Recalculate pairwise matrix from weights."""
    criteria = list(weights.keys())
    pairwise_matrix = np.zeros((len(criteria), len(criteria)))
    for i, c1 in enumerate(criteria):
        for j, c2 in enumerate(criteria):
            if c1 == c2:
                pairwise_matrix[i, j] = 1
            else:
                pairwise_matrix[i, j] = weights[c1] / weights[c2]
    return pairwise_matrix

def get_project_data(project_id):
    """Get all data related to a project."""
    try:
        project = get_project(project_id)
        if not project:
            return None

        # Get criteria and their types
        criteria = []
        criterion_types = []
        with open(CRITERIA_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['project_id'] == project_id:
                    criteria.append(row['name'])
                    criterion_types.append(row['type'])

        # Get weights if they exist
        weights = {}
        if os.path.exists(WEIGHTS_FILE):
            with open(WEIGHTS_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['project_id'] == project_id:
                        weights[row['criterion']] = float(row['weight'])

        # Get alternatives and their scores
        alternatives = []
        scores = {}
        with open(ALTERNATIVES_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['project_id'] == project_id:
                    alternatives.append(row['alternative'])
                    scores[row['alternative']] = {}

        with open(SCORES_FILE, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['project_id'] == project_id:
                    scores[row['alternative']][row['criterion']] = float(row['score'])

        # Get target profile if it exists (for Profile Matching)
        target_profile = {}
        if project['method'] == 'profile_matching' and os.path.exists(TARGET_PROFILES_FILE):
            with open(TARGET_PROFILES_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['project_id'] == project_id:
                        target_profile[row['criterion']] = int(row['target_value'])

        # Get final results
        results = []
        if os.path.exists(RESULTS_FILE):
            with open(RESULTS_FILE, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['project_id'] == project_id:
                        results.append({
                            'alternative': row['alternative'],
                            'final_score': float(row['final_score']),
                            'rank': int(row['rank'])
                        })

        return {
            'project': project,
            'criteria': criteria,
            'criterion_types': criterion_types,
            'weights': weights,
            'alternatives': alternatives,
            'scores': scores,
            'target_profile': target_profile,
            'results': sorted(results, key=lambda x: x['rank'])
        }
    except Exception as e:
        print(f"Error retrieving project data for project_id {project_id}: {e}", flush=True)
        return None

def calculate_results(project_id):
    """Calculate final results based on the project's method."""
    data = get_project_data(project_id)
    if not data:
        return None
    
    method = data['project']['method']
    alternatives = data['alternatives']
    criteria = data['criteria']
    
    # Calculate results based on method
    if method == 'ahp':
        # Use the AHP module to calculate results
        final_scores = ahp.calculate_ahp_results(project_id, criteria, alternatives)
    
    elif method == 'topsis':
        # For TOPSIS, use the scores from the form
        scores = data['scores']
        score_matrix = np.zeros((len(alternatives), len(criteria)))
        for i, alt in enumerate(alternatives):
            for j, crit in enumerate(criteria):
                score_matrix[i, j] = scores[alt][crit]
        
        weights = np.array([data['weights'][c] for c in criteria])
        criterion_types = data['criterion_types']
        final_scores = topsis.calculate_scores(score_matrix, weights, criterion_types)
    
    else:  # profile_matching
        scores = data['scores']
        target_values = np.array([data['target_profile'][c] for c in criteria])
        criterion_types = data['criterion_types']
        core_indices = [i for i, t in enumerate(criterion_types) if t == 'core']
        secondary_indices = [i for i, t in enumerate(criterion_types) if t == 'secondary']
        
        # Initialize score matrix with actual scores
        score_matrix = np.zeros((len(alternatives), len(criteria)))
        for i, alt in enumerate(alternatives):
            for j, crit in enumerate(criteria):
                score_matrix[i, j] = scores[alt][crit]
        
        # Calculate final scores using profile matching
        final_scores = profile_matching.calculate_scores(
            score_matrix,      # alternatives matrix (actual scores)
            target_values,     # target profile values
            core_indices,      # core criteria indices
            secondary_indices,  # secondary criteria indices
            core_weight=0.6,    # weight for core factors
            secondary_weight=0.4  # weight for secondary factors
        )
    
    # Sort alternatives by score and assign ranks
    results = []
    for alt, score in zip(alternatives, final_scores):
        results.append({
            'alternative': alt,
            'final_score': float(score)
        })
    
    # Sort by score (descending) and assign ranks
    results.sort(key=lambda x: x['final_score'], reverse=True)
    for i, result in enumerate(results, 1):
        result['rank'] = i
    
    # Read existing results
    existing_results = []
    with open(RESULTS_FILE, 'r', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['project_id'] != project_id:
                existing_results.append(row)
    
    # Add new results
    for result in results:
        existing_results.append({
            'project_id': project_id,
            'alternative': result['alternative'],
            'final_score': result['final_score'],
            'rank': result['rank']
        })
    
    # Write back all results
    with open(RESULTS_FILE, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['project_id', 'alternative', 'final_score', 'rank'])
        writer.writeheader()
        writer.writerows(existing_results)
    
    return results

# Routes
@app.route('/delete-project/<project_id>', methods=['POST'])
def delete_project(project_id):
    csv_handler = CSVHandler()
    csv_handler.delete_project(project_id)
    return jsonify({'success': True})

@app.route('/edit-project/<project_id>')
def edit_project(project_id):
    # Get project and all related data
    print('project_id', project_id)
    project_data = get_project_data(project_id)
    if not project_data:
        return redirect(url_for('dashboard'))
    
    # Store all project data in session
    session['project_id'] = project_id
    session['edit_mode'] = True
    session['project'] = project_data['project']
    session['criteria'] = project_data['criteria']
    session['criterion_types'] = project_data['criterion_types']
    session['alternatives'] = project_data['alternatives']
    session['scores'] = project_data['scores']
    session['weights'] = load_weights_from_csv(project_id)
    session['target_profile'] = project_data.get('target_profile', [])
    
    # Redirect to appropriate wizard
    return redirect(url_for('wizard.project_wizard', step=1, method=project_data['project']['method'].lower()))

@app.route('/dashboard')
def dashboard():
    """Display the main dashboard with all projects."""
    projects = get_projects()
    return render_template('dashboard.html', projects=projects)

@app.route('/method-selection')
def method_selection():
    """Display the method selection page."""
    return render_template('method_selection.html')

@app.route('/project-wizard/<method>')
def project_wizard(method):
    """Start the project wizard for a specific method."""
    if method not in ['ahp', 'topsis', 'profile_matching']:
        return redirect(url_for('method_selection'))
    
    # Clear any existing session data
    session.clear()
    session['method'] = method
    print('Starting project wizard for method:', session)
    return redirect(url_for('wizard.project_wizard', step=1, method=method))

@app.route('/calculate/<project_id>', methods=['POST'])
def calculate(project_id):
    """Calculate and save results for a project."""
    results = calculate_results(project_id)
    if not results:
        return jsonify({'error': 'Project not found'}), 404
    
    return redirect(url_for('results', project_id=project_id))

@app.route('/results/<project_id>')
def results(project_id):
    """Display the results page for a project."""
    data = get_project_data(project_id)
    if not data:
        return redirect(url_for('dashboard'))
    
    # If no results exist yet, calculate them
    if not data.get('results'):
        calculate_results(project_id)
        data = get_project_data(project_id)  # Refresh data
    
    return render_template('results.html',
                          project=data.get('project'),
                          criteria=data.get('criteria', []),
                          criterion_types=data.get('criterion_types', {}),
                          weights=data.get('weights', {}),
                          alternatives=data.get('alternatives', []),
                          scores=data.get('scores', {}),
                          results=data.get('results', []))

@app.route('/export/<project_id>')
def export_results(project_id):
    """Export project results as CSV."""
    data = get_project_data(project_id)
    if not data:
        return redirect(url_for('dashboard'))
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Write headers
    writer.writerow(['Project', data['project']['title']])
    writer.writerow(['Method', data['project']['method'].upper()])
    writer.writerow(['Created At', data['project']['created_at']])
    writer.writerow([])
    
    # Write criteria
    writer.writerow(['Criteria', 'Type', 'Weight'])
    for criterion, c_type in zip(data['criteria'], data['criterion_types']):
        weight = data['weights'].get(criterion, '-')
        writer.writerow([criterion, c_type, weight])
    writer.writerow([])
    
    # Write results
    writer.writerow(['Rank', 'Alternative', 'Final Score'])
    for result in data['results']:
        writer.writerow([result['rank'], result['alternative'], result['final_score']])
    
    # Prepare response
    output.seek(0)
    return Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=results_{project_id}.csv'
        }
    )

@app.route('/export-pdf/<project_id>')
def export_results_pdf(project_id):
    """Export project results as a PDF with graphics."""
    data = get_project_data(project_id)
    if not data:
        return redirect(url_for('dashboard'))

    # Create a PDF file
    pdf_path = os.path.join(DATA_DIR, f'results_{project_id}.pdf')
    c = canvas.Canvas(pdf_path, pagesize=letter)
    width, height = letter

    # Add title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(100, height - 50, f"Project: {data['project']['title']}")

    # Add project details
    c.setFont("Helvetica", 12)
    c.drawString(100, height - 80, f"Method: {data['project']['method'].upper()}")
    c.drawString(100, height - 100, f"Created At: {data['project']['created_at']}")

    # Generate Rankings Chart
    rankings_chart_path = os.path.join(DATA_DIR, f'rankings_chart_{project_id}.png')
    plt.bar(
        [result['alternative'] for result in data['results']],
        [result['final_score'] for result in data['results']],
        color=['green' if result['rank'] == 1 else 'blue' for result in data['results']]
    )
    plt.title('Alternative Rankings')
    plt.xlabel('Alternatives')
    plt.ylabel('Scores')
    plt.savefig(rankings_chart_path)
    plt.close()

    # Add Rankings Chart to PDF
    c.drawImage(rankings_chart_path, 100, height - 300, width=400, height=200)

    # Generate Criteria Weights Chart (if applicable)
    if data['weights']:
        weights_chart_path = os.path.join(DATA_DIR, f'weights_chart_{project_id}.png')
        plt.pie(
            data['weights'].values(),
            labels=data['weights'].keys(),
            autopct='%1.1f%%',
            startangle=140
        )
        plt.title('Criteria Weights')
        plt.savefig(weights_chart_path)
        plt.close()

        # Add Criteria Weights Chart to PDF
        c.drawImage(weights_chart_path, 100, height - 550, width=400, height=200)

    # Adjust the position of the summary to avoid overlapping with the graphics
    summary_y_position = height - 600 if data['weights'] else height - 350

    # Add summary of the first ranking alternative
    top_alternative = data['results'][0]  # Assuming results are sorted by rank
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, summary_y_position, "Summary:")
    c.setFont("Helvetica", 12)
    c.drawString(120, summary_y_position - 20, f"Top Alternative: {top_alternative['alternative']}")
    c.drawString(120, summary_y_position - 40, f"Score: {top_alternative['final_score']}")

    # Save the PDF
    c.save()

    # Delete the temporary chart files after adding them to the PDF
    if os.path.exists(rankings_chart_path):
        os.remove(rankings_chart_path)
    if data['weights'] and os.path.exists(weights_chart_path):
        os.remove(weights_chart_path)

    # Return the PDF as a response
    return send_file(pdf_path, as_attachment=True)

if __name__ == '__main__':
    init_csv_files()
    app.run(debug=True, port=5001)
