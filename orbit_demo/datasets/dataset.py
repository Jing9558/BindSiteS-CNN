"""
orbit.datasets.dataset

"""
import urllib.request as request
import requests
import zipfile
import tarfile
import abc
import os

from loguru import logger
from orbit.config import DATA_DIR

class BindingSiteDataset(abc.ABC):

    def __init__(self, db_name):
        root = os.environ.get(DATA_DIR)
        if not root:
            raise ValueError(f'Environment variable: {DATA_DIR} not set')
        self._root = os.path.join(root, db_name)
        self._db_name = db_name
        self._size = None

    @property
    def root(self):
        return self._root

    @property
    def processed_dir(self):
        return os.path.join(self.root, 'processed/orbit')

    @abc.abstractmethod
    def get_entries(self, include_metadata=True):
        ...

    @abc.abstractmethod
    def download(self):
        ...

    @abc.abstractmethod
    def preprocess(self):
        ...

    def _download_file(self, fname, url):
        logger.info(f'Downloading file: {fname} from {url}')
        path = os.path.join(self.root, fname)
        request.urlretrieve(url, path)

    def _download_tar(self, fname, url):
        logger.info(f'Downloading file: {fname} from {url}')
        path = os.path.join(self.root, fname)
        request.urlretrieve(url, path)
        archive = tarfile.open(path)
        archive.extractall(os.path.dirname(path))
        archive.close()
        os.remove(path)

    def _download_zip(self, fname, url):
        logger.info(f'Downloading file: {fname} from {url}')
        path = os.path.join(self.root, fname)
        r = requests.get(url)
        r.raise_for_status()
        with open(path, 'wb') as result:
            result.write(r.content)
        with zipfile.ZipFile(path, 'r') as archive:
            archive.extractall(self.root)
        os.remove(path)

    def __len__(self):
        if self._size:
            return self._size
        try:
            self._size = len(self.get_entries(include_metadata=False))
        except Exception:
            return 0
        return self._size

    def __repr__(self):
        return '{}(entries: {})'.format(
            self.__class__.__name__,
            len(self)
        )
