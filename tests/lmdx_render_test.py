"""Tests for rendering Gemini LMDX suggestion blocks into readable Markdown."""
from types import SimpleNamespace

from gemini_openai import server


def test_render_lmdx_elicitations_group_and_followup():
    raw = (
        "Intro text.\n\n"
        '<ElicitationsGroup message="¿Por dónde quieres continuar ahora?">\n'
        '  <Elicitation label="Cachear toolchain" query="Add actions/cache for toolchain." />\n'
        '  <Elicitation label="Revisar .wgt" query="Revisar .wgt" />\n'
        "</ElicitationsGroup>\n"
        '<FollowUp label="Ver logs" query="Inspect logs/server.log" />'
    )
    rendered = server._render_lmdx(raw)
    assert "<ElicitationsGroup" not in rendered
    assert "<Elicitation" not in rendered
    assert "<FollowUp" not in rendered
    assert "**¿Por dónde quieres continuar ahora?**" in rendered
    assert "1. **Cachear toolchain** — `Add actions/cache for toolchain.`" in rendered
    assert "2. **Revisar .wgt**" in rendered
    assert "**Siguiente paso sugerido:** Ver logs (`Inspect logs/server.log`)" in rendered


def test_render_output_converts_lmdx_blocks():
    out = SimpleNamespace(
        text='<FollowUp label="Continuar" query="Run next step" />',
        images=[],
        videos=[],
    )
    rendered = server.render_output(out)
    assert "<FollowUp" not in rendered
    assert "Continuar" in rendered
    assert "Run next step" in rendered
