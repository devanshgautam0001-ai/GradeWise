"""
Views package for Student Performance Prediction and Analysis System
"""

from views.overview import render_overview_view
from views.dataset_management import render_dataset_management_view
from views.prediction import render_prediction_view
from views.risk_analysis import render_risk_analysis_view
from views.tree_explorer import render_tree_explorer_view
from views.model_evaluation import render_model_evaluation_view
from views.what_if_simulator import render_what_if_simulator_view
from views.recommendations import render_recommendations_view
from views.history import render_history_view
from views.about import render_about_view

__all__ = [
    "render_overview_view",
    "render_dataset_management_view",
    "render_prediction_view",
    "render_risk_analysis_view",
    "render_tree_explorer_view",
    "render_model_evaluation_view",
    "render_what_if_simulator_view",
    "render_recommendations_view",
    "render_history_view",
    "render_about_view",
]
