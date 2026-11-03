"""Lightweight studio entry point: python server.py (no Gradio or torch imports)."""
import os

from src.core.pipeline import build_context
from src.ui.web import create_web_app
from src.utils.logging_config import setup_logging


def create_app():
    setup_logging()
    return create_web_app(build_context())


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(create_app(), host=os.environ.get('MEDISCAN_HOST', '127.0.0.1'),
                port=int(os.environ.get('MEDISCAN_PORT', '7860')), workers=1)
