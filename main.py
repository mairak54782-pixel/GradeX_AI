"""Run this file to start GradEx AI: python main.py"""

from app import build_app
from config import get_settings

if __name__ == "__main__":
    settings = get_settings()
    settings.vector_db_path.mkdir(parents=True, exist_ok=True)
    demo = build_app(settings)
    demo.launch()
