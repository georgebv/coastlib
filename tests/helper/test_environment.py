import os
from coastlib.helper.environment import append_bin


def test_append_bin():
    append_bin()
    assert os.path.join('coastlib', 'bin') in os.environ['PATH']
    bin_path = None
    for path_value in os.environ['PATH'].split(os.pathsep):
        if os.path.join('coastlib', 'bin') in path_value:
            bin_path = path_value
            break
    assert os.path.exists(os.path.join(bin_path, 'Fourier.exe'))
