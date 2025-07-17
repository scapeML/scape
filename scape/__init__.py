import os
# Set JAX backend for Keras before any imports
os.environ.setdefault('KERAS_BACKEND', 'jax')

import scape._model as model
import scape._losses as losses
import scape._io as io
import scape._util as util
from scape._model import SCAPE

__version__ = "0.1.1"
