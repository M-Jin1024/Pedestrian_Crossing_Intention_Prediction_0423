import os
import random

import numpy as np


def set_global_determinism(seed=42):
    """
    Fixes the random state across Python, NumPy and TensorFlow so that
    subsequent training/evaluation runs are reproducible.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('TF_DETERMINISTIC_OPS', '1')
    os.environ.setdefault('TF_CUDNN_DETERMINISTIC', '1')

    random.seed(seed)
    np.random.seed(seed)

    try:
        import tensorflow as tf
    except ImportError:  # pragma: no cover
        return

    tf.random.set_seed(seed)

    try:
        tf.keras.utils.set_random_seed(seed)
    except AttributeError:
        pass

    try:
        tf.config.experimental.enable_op_determinism()
    except AttributeError:
        pass

    tf.keras.backend.clear_session()
