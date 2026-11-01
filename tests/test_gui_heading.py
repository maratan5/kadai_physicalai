"""Regression test for the large in-window heading."""
from pathlib import Path


def test_header_displays_pi_0_5_with_explicit_font():
    source = (Path(__file__).parents[1] / 'kadai_physicalai' / 'gui.py').read_text(encoding='utf-8')
    assert "text='π0.5  |  Physical AI  |  Galaxea R1 Pro'" in source
    assert "font=('DejaVu Sans', 19, 'bold')" in source
    assert "text='π0.0" not in source
