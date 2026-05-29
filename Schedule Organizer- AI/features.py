import numpy as np

def featurize_task(text):
    if not text:
        text = ""
    words = text.split()
    wc = len(words)
    cc = len(text)
    return np.array([wc, cc], dtype=float)
