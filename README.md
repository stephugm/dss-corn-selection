# Decision Support System

A web-based Decision Support System implementing multiple decision-making methods (AHP, TOPSIS, and Profile Matching).

## Features

- Multiple decision-making methods:
  - Analytic Hierarchy Process (AHP)
  - Technique for Order Preference by Similarity to Ideal Solution (TOPSIS)
  - Profile Matching (PM)
- Modern, responsive UI built with TailwindCSS
- Project management with CRUD operations
- Interactive wizards for each method
- Results visualization using Chart.js

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\Activate.ps1
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your browser and navigate to `http://localhost:5001`

## Project Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── static/            # Static assets (CSS, JS)
└── templates/         # HTML templates
    ├── base.html     # Base template with common layout
    ├── dashboard.html
    ├── method_selection.html
    └── project_wizard.html
```

## Methods Implementation

### AHP
- Pairwise comparison matrices
- Consistency ratio calculation
- Priority vector computation

### TOPSIS
- Decision matrix normalization
- Weighted normalized decision matrix
- Ideal and negative-ideal solutions
- Relative closeness calculation

### Profile Matching
- Gap analysis
- Core and secondary factors
- Weighted total calculation
