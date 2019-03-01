import coastlib.bin
import os


def append_bin():
    """
    Appends coastlib/bin to system environment Path variable.
    Allows calling binary files from the terminal.
    """

    bin_path = os.path.split(str(coastlib.bin.__file__))[0]
    os.environ['PATH'] += os.pathsep + bin_path

