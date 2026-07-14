"""Sanity that tensor-layouts is usable in this environment."""

from tensor_layouts import Layout, compose, logical_divide


def test_basic_layout():
    layout = Layout((4, 8), (1, 4))
    assert layout(2, 3) == 14


def test_compose_and_divide():
    a = Layout((4, 2), (1, 4))
    b = Layout((2, 4), (4, 1))
    c = compose(a, b)
    assert c is not None
    divided = logical_divide(Layout(16, 1), 4)
    assert divided is not None
