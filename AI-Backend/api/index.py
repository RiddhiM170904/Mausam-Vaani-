"""
Vercel serverless entry point
Imports the FastAPI app from parent directory
"""
import sys
import os

# Add parent directory to path to import app
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from app import app

# Vercel expects 'app' or 'handler'
# FastAPI works directly with Vercel's ASGI adapter
