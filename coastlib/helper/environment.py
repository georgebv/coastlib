import coastlib.bin
import os


def append_bin():
    """
    Appends coastlib/bin to system environment PATH variable.
    Allows calling binary files from the terminal.
    """

    bin_path = os.path.split(str(coastlib.bin.__file__))[0]
    if bin_path not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + bin_path
