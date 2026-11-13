from __future__ import annotations

from flask import Flask

from .model import DiseasePredictor, create_predictor, predict_disease


def create_app() -> Flask:
    """Load the Flask factory from the project's root application module."""
    from app import create_app as create_flask_app

    return create_flask_app()


__all__ = ["DiseasePredictor", "create_app", "create_predictor", "predict_disease"]
