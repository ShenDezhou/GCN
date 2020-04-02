from __future__ import absolute_import, division, unicode_literals

import re

from . import base
from ..constants import rcdataElements, spaceCharacters
spaceCharacters = "".join(spaceCharacters)

SPACES_REGEX = re.compile("[%s]+" % spaceCharacters)


class Filter(base.Filter):
    """Collapses whitespace except in pre, textarea, and