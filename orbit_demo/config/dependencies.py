"""
orbit.config.dependencies
Configuration of external (non-Python) dependencies.

"""
import os

from collections.abc import MutableMapping
from loguru import logger

from orbit.config.config_utils import which


class ExternalDependencyError(Exception):
    """Exception raised for missing external dependencies."""

    def __init__(self, program: str, url: str):
        message = f'Missing external dependency: "{program}". '
        message += f'The software can be download from: {url}'
        super(ExternalDependencyError, self).__init__(message)


class Dependency(object):

    def __init__(self, path, url=None, soft=False):
        self._path, self._name = self._process(path)
        self._url = url
        self._soft = soft

    @staticmethod
    def _process(path):
        _, fname = os.path.split(path)
        fname, _ = os.path.splitext(fname)
        return which(path), fname

    @property
    def name(self):
        return self._name

    @property
    def url(self):
        return self._url

    @property
    def is_required(self):
        return self._soft is False

    def get_path(self):
        if self._path is None and self.is_required:
            raise ExternalDependencyError(self.name, self.url)
        return self._path

    def __repr__(self):
        return '{}({}, path={}, required={})'.format(
            self.__class__.__name__,
            self._name,
            self._path,
            self.is_required
        )


class DependencyManager(MutableMapping):

    def __init__(self):
        super(DependencyManager, self).__init__()

    def register(self, path, url=None, required=True):
        dep = Dependency(path, url, required is False)
        self[dep.name] = dep

    def is_registered(self, name):
        return name in self

    def test(self, raise_on_error=False):
        status = 0
        for k, v in self.items():
            try:
                path = v.get_path()
                if path is None:
                    logger.warning(
                        f"Could not locate soft external dependency: {k}"
                    )
                else:
                    logger.info(
                        f"Successfully located external dependency: {k}"
                    )
            except ExternalDependencyError as e:
                status = 1
                logger.error(
                    f"Could not locate required external dependency: {k}"
                )
                if raise_on_error:
                    raise e
        return status

    def __getitem__(self, key):
        return self.__dict__[key]

    def __setitem__(self, key, value):
        if not isinstance(value, Dependency):
            raise ValueError('')
        self.__dict__[key] = value

    def __delitem__(self, key):
        del self.__dict__[key]

    def __iter__(self):
        return iter(self.__dict__)

    def __len__(self):
        return len(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return '{}({})'.format(
            self.__class__.__name__,
            self.__dict__
        )
