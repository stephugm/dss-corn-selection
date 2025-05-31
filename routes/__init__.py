"""
Routes package for the Decision Support System.
"""
from flask import Blueprint

wizard_bp = Blueprint('wizard', __name__, url_prefix='')

from . import ahp_routes, topsis_routes, profile_matching_routes, common_routes
