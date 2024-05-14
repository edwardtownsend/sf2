import pytest
import numpy as np
from matplotlib import pyplot as plt

from cued_sf2_lab.familiarisation import (
    load_mat_img, plot_image, multiples_pow2_between
)

def test_load_mat_img_1():
    """
    Test for an invalid image name.
    """
    img = 'wrong_img_name'
    img_info = 'X'
    cmap_info = {'map', 'map2'}
    with pytest.raises(ValueError):
        load_mat_img(img, img_info, cmap_info)


def test_load_lighthouse():
    img, cmaps = load_mat_img(
        'lighthouse.mat', img_info='X', cmap_info={'map', 'map2'})
    assert img.dtype == np.uint8
    assert img.shape == (256, 256)
    assert cmaps['map'] is not None
    assert cmaps['map2'] is not None


class TestMultiplesPow2Between:

    def test_aligned_integer(self):
        mp2b = multiples_pow2_between
        np.testing.assert_equal(mp2b(0, 16, n=5), [0, 4, 8, 12, 16])
        np.testing.assert_equal(mp2b(0, 16, n=4), [0, 8, 16])
        np.testing.assert_equal(mp2b(0, 16, n=3), [0, 8, 16])
        np.testing.assert_equal(mp2b(0, 16, n=2), [0, 16])
        np.testing.assert_equal(mp2b(0, 16, n=1), [0])
        np.testing.assert_equal(mp2b(0, 16, n=0), [0])

    def test_unaligned_integer(self):
        mp2b = multiples_pow2_between
        np.testing.assert_equal(mp2b(-5, 15, n=5), [-4, 0, 4, 8, 12])

    def test_aligned_fractional(self):
        mp2b = multiples_pow2_between
        np.testing.assert_equal(mp2b(0, 1, n=5), [0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_equal(mp2b(0, 1, n=4), [0, 0.5, 1])
        np.testing.assert_equal(mp2b(0, 1, n=3), [0, 0.5, 1])
        np.testing.assert_equal(mp2b(0, 1, n=2), [0, 1])
        np.testing.assert_equal(mp2b(0, 1, n=1), [0])

    def test_unaligned_fractional(self):
        mp2b = multiples_pow2_between
        np.testing.assert_equal(mp2b(-0.1, 1.1, n=5), [0, 0.25, 0.5, 0.75, 1])
        np.testing.assert_equal(mp2b(-0.1, 1.1, n=4), [0, 0.5, 1])
        np.testing.assert_equal(mp2b(-0.1, 1.1, n=3), [0, 0.5, 1])
        np.testing.assert_equal(mp2b(-0.1, 1.1, n=2), [0, 1])
        np.testing.assert_equal(mp2b(-0.1, 1.1, n=1), [0])


def test_plot_image(tmp_path):
    # we don't bother testing the plot looks ok
    img, _ = load_mat_img('lighthouse.mat', img_info='X', cmap_info={})
    fig, ax = plt.subplots()
    im_obj = plot_image(img, ax=ax)
    fig.colorbar(im_obj)
    fig.savefig(str(tmp_path / 'lighthouse-plot.png'))
    plt.close(fig)
