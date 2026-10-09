"""A scientific failure must remain a reportable outcome."""

import json

from analysis.research_campaign import causal_report


def test_all_failed_references_produce_negative_gate(tmp_path):
    report = causal_report([], {}, tmp_path)
    assert report == {"gate": "G2", "passed": False, "reason": "no_complete_pairs"}
    assert json.loads((tmp_path / "gate_g2.json").read_text()) == report
