"""Smoke test: config + state import cleanly."""
def test_config_imports():
    from app.config import settings
    assert settings.EMBED_DIM == 768
    assert settings.LLM_MODEL


def test_state_imports():
    from app.graph_state import RAGState
    assert "question" in RAGState.__annotations__
