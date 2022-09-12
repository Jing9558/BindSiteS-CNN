"""
orbit.config.config_utils
Configuration utilities.

"""
import typing
import sys
import os


def is_executable(path: str) -> bool:
    """bool: return True if path is executable."""
    return (
        isinstance(path, str) and
        os.path.exists(path) and
        os.access(path, os.X_OK)
    )


def which(program: str) -> typing.Union[str, None]:
    """Returns the path to a specified program.

    Parameters
    ----------
    program
        Path to a program or the name of a program

    Returns
    -------
    path
        Path to the program if exists and is
        executable else None.

    """
    fpath, fname = os.path.split(program)
    fname, fext = os.path.splitext(fname)
    if fname and is_executable(program):
        return program
    if os.name == 'nt' and fext == '':
        program += '.exe'
    env_paths = os.environ['PATH'].split(os.pathsep)
    env_paths.append(os.path.dirname(sys.executable))
    for path in env_paths:
        path = os.path.join(path, program)
        if is_executable(path):
            return path
    fname = fname.upper() + '_BIN'
    if fname in os.environ:
        return os.environ[fname]
    return None
