def test_deps_importable():
    import torch, numpy, matplotlib
    assert torch.__version__
    assert numpy.__version__
    assert matplotlib.__version__

def test_python_version():
    import sys
    assert sys.version_info >= (3, 9)
