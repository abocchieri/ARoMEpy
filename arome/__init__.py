import importlib.metadata as metadata
from datetime import date

from .__version__ import __version__

# load package info
__pkg_name__ = metadata.metadata("aromepy")["Name"]
__author__ = metadata.metadata("aromepy")["Author"]
__license__ = metadata.metadata("aromepy")["license"]
__copyright__ = "2023-{:d}, {}".format(date.today().year, __author__)
__summary__ = metadata.metadata("aromepy")["Summary"]
