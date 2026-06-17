import os
from loguru import logger


def init_phoenix(project_name: str = "rag-eval-framework") -> object:
    """
    Start Arize Phoenix locally and instrument all OpenAI calls.
    Returns the Phoenix session (has a .url attribute).
    Silently skips if Phoenix is not installed or port is in use.
    """
    try:
        import phoenix as px
        from phoenix.otel import register
        from openinference.instrumentation.openai import OpenAIInstrumentor

        session = px.launch_app()
        tracer_provider = register(project_name=project_name)
        OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
        logger.info(f"Arize Phoenix running at: {session.url}")
        return session
    except Exception as e:
        logger.warning(f"Arize Phoenix not started: {e}")
        return None
