"""Chest X-ray triage: training, evaluation and serving utilities.

Keep this module free of heavy imports: the serving image installs only numpy/opencv/onnxruntime
and imports `cxr.labels` and `cxr.preprocess` from here.
"""

__version__ = "0.1.0"
