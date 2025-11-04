"""
test.py — Comprehensive tests for your `convolve(image, kernel)` function.

How to use:
  1) Place this file alongside your implementation (e.g., MyConvolution.py).
  2) Make sure your module exposes:  def convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray
  3) Run:  python -m unittest test.py  (or)  python test.py

Notes:
  • Tests compare against a slow, known-correct reference implementation (zero-padding, TRUE convolution).
  • If SciPy is available, a few tests also cross-check against scipy.signal.convolve2d.
  • The coursework forbids library convs in your submission, but using them here for local testing is fine.
  • If your function does not assert odd-sized kernels, test_even_kernel_raises will fail (as per spec).
"""

import unittest
import numpy as np
from MyConvolution import convolve as reference_convolve
 # Adjust if your module name is different

# Try to import the student's convolve function.
# Adjust the module name below if yours is different.
_import_errors = []
convolve = None
for modname in ["MyConvolution", "myconvolution", "convolve", "main"]:
    try:
        mod = __import__(modname)
        if hasattr(mod, "convolve"):
            convolve = getattr(mod, "convolve")
            break
    except Exception as e:
        _import_errors.append((modname, repr(e)))
if convolve is None:
    raise ImportError(
        "Could not import `convolve` from MyConvolution/myconvolution/convolve/main. "
        f"Import attempts and errors: {_import_errors}. "
        "Rename your file to MyConvolution.py or edit the import list above."
    )

# Optional: SciPy check (used in a couple of tests if present).
_has_scipy = False
try:
    from scipy.signal import convolve2d as sp_convolve2d
    _has_scipy = True
except Exception:
    pass


def gaussian_kernel(sigma: float) -> np.ndarray:
    size = int(np.floor(8*sigma + 1))
    if size % 2 == 0:
        size += 1
    ax = np.arange(-(size//2), size//2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    K = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    K /= K.sum()
    return K


class TestConvolve(unittest.TestCase):

    # 1
    def test_identity_gray(self):
        img = np.arange(16, dtype=np.float64).reshape(4,4)
        I = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
        y = convolve(img, I).astype(np.float64)
        self.assertTrue(np.allclose(y, img))

    # 2
    def test_zero_kernel_gray(self):
        img = np.arange(16, dtype=np.float64).reshape(4,4)
        Z = np.zeros((3,3), dtype=np.float64)
        y = convolve(img, Z).astype(np.float64)
        self.assertEqual(np.count_nonzero(y), 0)

    # 3
    def test_identity_color(self):
        imgc = np.dstack([
            np.arange(16).reshape(4,4),
            np.arange(16,32).reshape(4,4),
            np.arange(32,48).reshape(4,4)
        ]).astype(np.float64)
        I = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float64)
        y = convolve(imgc, I).astype(np.float64)
        self.assertTrue(np.allclose(y, imgc))

    # 4
    def test_zero_kernel_color(self):
        imgc = np.random.default_rng(0).random((6,6,3))
        Z = np.zeros((5,5), dtype=np.float64)
        y = convolve(imgc, Z)
        self.assertEqual(np.count_nonzero(y), 0)

    # 5
    def test_asymmetric_flip_gray_center(self):
        img = np.arange(25, dtype=np.float64).reshape(5,5)
        K = np.array([[0,1,2],
                      [3,4,5],
                      [6,7,8]], dtype=np.float64)
        ref = reference_convolve(img, K)
        y = convolve(img, K).astype(np.float64)
        self.assertTrue(np.isclose(y[2,2], ref[2,2]))
        self.assertTrue(np.allclose(y, ref))

    # 6
    def test_box_blur_uniform_preserve_center(self):
        imgu = np.full((7,7), 10.0, dtype=np.float64)
        box = np.ones((3,3), dtype=np.float64) / 9.0
        y = convolve(imgu, box).astype(np.float64)
        self.assertTrue(np.allclose(y[1:-1,1:-1], 10.0))

    # 7
    def test_box_blur_border_effect(self):
        img = np.zeros((5,5), dtype=np.float64)
        img[2,2] = 1.0
        box = np.ones((3,3), dtype=np.float64) / 9.0
        y = convolve(img, box).astype(np.float64)
        # With zero padding, center is 1/9; corners also 1/9 since only one 1 overlaps
        self.assertTrue(np.isclose(y[2,2], 1/9))
        self.assertTrue(np.isclose(y[0,0], 1/9))

    # 8
    def test_random_gray_compare_reference(self):
        rng = np.random.default_rng(42)
        img = rng.random((32,32))
        K = rng.normal(size=(5,5))
        K /= np.sum(np.abs(K)) + 1e-12
        y = convolve(img, K).astype(np.float64)
        ref = reference_convolve(img, K)
        self.assertTrue(np.allclose(y, ref, atol=1e-10))

    # 9
    def test_random_color_compare_reference(self):
        rng = np.random.default_rng(1)
        img = rng.random((16,16,3))
        K = rng.normal(size=(3,3))
        y = convolve(img, K).astype(np.float64)
        ref = reference_convolve(img, K)
        self.assertTrue(np.allclose(y, ref, atol=1e-10))

    # 10
    def test_non_square_kernel(self):
        img = np.arange(36, dtype=np.float64).reshape(6,6)
        K = np.array([[0,1,0,0,1],
                      [1,0,1,0,0],
                      [0,1,0,1,0]], dtype=np.float64)  # 3x5 odd-sized
        y = convolve(img, K).astype(np.float64)
        ref = reference_convolve(img, K)
        self.assertTrue(np.allclose(y, ref))

    # 11
    def test_large_kernel_on_small_image(self):
        img = np.arange(16, dtype=np.float64).reshape(4,4)
        K = np.ones((7,7), dtype=np.float64) / 49.0
        y = convolve(img, K)
        ref = reference_convolve(img, K)
        self.assertTrue(np.allclose(y, ref))

    # 12
    def test_uint8_roundtrip_clip(self):
        rng = np.random.default_rng(0)
        img = rng.integers(0, 255, size=(20,20), dtype=np.uint8)
        K = np.array([[1,2,1],[2,4,2],[1,2,1]], dtype=np.float64)
        K = K / K.sum()
        y = convolve(img, K)
        ref = reference_convolve(img, K)
        # Compare after clipping/casting to uint8 like a typical implementation
        ref_uint8 = np.clip(ref, 0, 255).astype(np.uint8)
        self.assertEqual(y.dtype, np.uint8)
        self.assertTrue(np.allclose(y, ref_uint8))

    # 13
    def test_float32_dtype_preserved_or_cast_reasonably(self):
        img = np.random.default_rng(3).random((8,8)).astype(np.float32)
        K = gaussian_kernel(1.0).astype(np.float32)
        y = convolve(img, K)
        # Implementation may upcast to float64 internally; allow float32 or float64
        self.assertIn(y.dtype, (np.float32, np.float64))
        ref = reference_convolve(img.astype(np.float64), K.astype(np.float64))
        self.assertTrue(np.allclose(y, ref, atol=1e-6))

    # 14
    def test_even_kernel_raises(self):
        img = np.zeros((5,5), dtype=np.float64)
        K = np.ones((4,4), dtype=np.float64)
        with self.assertRaises(AssertionError, msg="Spec requires odd-sized kernels; add an assertion."):
            _ = convolve(img, K)

    # 15
    def test_gaussian_kernel_normalized_effect(self):
        img = np.full((9,9), 100.0)
        K = gaussian_kernel(1.2)
        y = convolve(img, K).astype(np.float64)
        # Interior should remain ~100 due to normalization
        self.assertTrue(np.allclose(y[2:-2,2:-2], 100.0, atol=1e-12))

    # 16
    def test_compare_with_scipy_gray_if_available(self):
        if not _has_scipy:
            self.skipTest("SciPy not available")
        img = np.random.default_rng(0).random((20,20))
        K = np.array([[0,1,0],[1,4,1],[0,1,0]], dtype=np.float64) / 8.0
        y = convolve(img, K).astype(np.float64)
        sp = sp_convolve2d(img, K, mode="same", boundary="fill", fillvalue=0)
        self.assertTrue(np.allclose(y, sp, atol=1e-12))

    # 17
    def test_compare_with_scipy_color_if_available(self):
        if not _has_scipy:
            self.skipTest("SciPy not available")
        img = np.random.default_rng(2).random((15,15,3))
        K = gaussian_kernel(0.8)
        y = convolve(img, K).astype(np.float64)
        spc = np.dstack([sp_convolve2d(img[...,c], K, mode="same", boundary="fill", fillvalue=0)
                         for c in range(3)])
        self.assertTrue(np.allclose(y, spc, atol=1e-12))

    # 18
    def test_highpass_negative_values(self):
        rng = np.random.default_rng(4)
        img = rng.random((16,16))
        K = gaussian_kernel(1.0)
        blur = convolve(img, K).astype(np.float64)
        hp = img - blur
        # Mean of high-pass tends toward 0
        self.assertTrue(abs(hp.mean()) < 1e-6)

    # 19
    def test_three_by_five_kernel_gray(self):
        img = np.arange(30, dtype=np.float64).reshape(5,6)
        K = np.array([[1,0,-1,0,1],
                      [2,0,-2,0,2],
                      [1,0,-1,0,1]], dtype=np.float64)
        y = convolve(img, K).astype(np.float64)
        ref = reference_convolve(img, K)
        self.assertTrue(np.allclose(y, ref))

    # 20
    def test_many_random_cases(self):
        rng = np.random.default_rng(123)
        for _ in range(5):  # five combos = 5 additional cases
            H, W = rng.integers(8, 24, size=2)
            C = rng.integers(1, 4)
            kh = int(rng.choice([3,5,7]))
            kw = int(rng.choice([3,5,7]))
            img = rng.random((H,W,C)) if C>1 else rng.random((H,W))
            K = rng.normal(size=(kh,kw))
            y = convolve(img, K).astype(np.float64)
            ref = reference_convolve(img, K)
            self.assertTrue(np.allclose(y, ref, atol=1e-10))


if __name__ == "__main__":
    unittest.main()
