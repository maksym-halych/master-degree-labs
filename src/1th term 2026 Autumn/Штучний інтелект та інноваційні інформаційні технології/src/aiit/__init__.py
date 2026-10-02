"""Code shared by the labs of this subject.

The entry points are the lab*/run.py scripts; all the computation lives here.
"""

import os

# Must be set before the first `import tensorflow` in any module, otherwise
# TensorFlow has already printed its informational messages. The package is
# imported ahead of any of its own modules, which makes this the most reliable
# place: in tf_models.py the variable was set too late, because set_seeds() in
# run_lab1 imported tensorflow first.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
