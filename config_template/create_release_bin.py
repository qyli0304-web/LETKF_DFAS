"""
Utility script to generate a uniform float32 binary file for release factors.
The output file can be used as a temporal factor input by external models.
"""
import numpy as np

arr = np.ones(1000, dtype=np.float32)
arr.astype(np.float32).tofile("config_template/uniform_Cs137_30min.bin")
