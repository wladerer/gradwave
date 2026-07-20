"""Regression tests for pseudopotential-parser review fixes."""

from __future__ import annotations

from gradwave.pseudo.upf import _read_root


def test_read_root_preserves_valid_entities_during_salvage(tmp_path):
    """The XML-salvage path must escape only *bare* '&' — legitimate entities
    like &lt; elsewhere in the file must survive, not get double-escaped to
    &amp;lt;. A bare '&' forces the ParseError → salvage retry.
    """
    text = (
        '<UPF version="2.0.1">\n'
        "  <PP_INFO>generator notes with a stray & and < junk</PP_INFO>\n"
        "  <PP_NOTE>convergence &lt; 1e-6 reached &amp; stable, R&D grade</PP_NOTE>\n"
        "</UPF>\n"
    )
    path = tmp_path / "salvage.upf"
    path.write_text(text)

    root = _read_root(path)  # must not raise
    note = root.find("PP_NOTE").text

    # &lt; decoded to a real '<' (not the literal "&lt;" of a double-escape)
    assert "convergence < 1e-6" in note
    assert "&lt;" not in note
    # &amp; decoded to a single '&', and the bare "R&D" '&' round-tripped once
    assert "reached & stable" in note
    assert "R&D grade" in note
    assert "&amp;" not in note
