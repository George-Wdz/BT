import sys

import numpy as np

from dataset import load_npz


def test_numpy_core_compatibility_aliases(tmp_path):
    path = tmp_path / "sample.npz"
    samples = np.asarray([{"features": np.ones((1, 2), dtype=np.float32)}], dtype=object)
    np.savez_compressed(path, samples=samples, splits=np.asarray(["train"]))

    loaded, splits = load_npz(str(path))

    assert loaded[0]["features"].shape == (1, 2)
    assert splits.tolist() == ["train"]
    if not hasattr(np, "_core"):
        assert sys.modules["numpy._core"] is np.core
