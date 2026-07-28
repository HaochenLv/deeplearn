import os
from run import plot_extrapolation, plot_forgetting


def test_plot_extrapolation_writes_file(tmp_path):
    out = tmp_path / "extrap.png"
    plot_extrapolation([1, 2, 3, 4, 5, 6, 7, 8],
                       [1.0, 0.99, 0.98, 0.97, 0.9, 0.7, 0.4, 0.2],
                       str(out))
    assert os.path.getsize(str(out)) > 0


def test_plot_forgetting_writes_file(tmp_path):
    out = tmp_path / "forget.png"
    history = [(1, 0.95, {1: 0.95, 2: 0.1}),
               (2, 0.96, {1: 0.94, 2: 0.96}),
               (3, 0.95, {1: 0.93, 2: 0.95, 3: 0.95})]
    plot_forgetting(history, [1, 2, 3], str(out))
    assert os.path.getsize(str(out)) > 0
