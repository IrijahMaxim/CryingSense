"""
Audio Test Module for CryingSense

This module provides tools for testing the CryingSense CNN model with live audio.
"""

__version__ = "1.0.0"
__author__ = "CryingSense Team"

from .record_audio import AudioRecorder
from .test_live import CryingSensePredictor

__all__ = ['AudioRecorder', 'CryingSensePredictor']
