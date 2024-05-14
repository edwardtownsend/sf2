import pytest
import numpy as np

from cued_sf2_lab.bitword import bitword


def test_access():
    b = bitword(0b0101, 4)
    assert b.val == 0b0101
    assert b.bits == 4


def test_guess():
    assert bitword(0).bits == 0
    assert bitword(0b0101).bits == 3


def test_invalid():
    with pytest.raises(OverflowError):
        bitword(0b11, 1)


def test_str():
    assert str(bitword(0b0101, 4)) == "0b0101"
    assert str(bitword(0, 4)) == "0b0000"
    assert str(bitword(0, 1)) == "0b0"


def test_repr():
    assert repr(bitword(0b0101, 4)) == "bitword(0b0101, 4)"
    assert repr(bitword(0, 4)) == "bitword(0b0000, 4)"
    assert repr(bitword(0, 1)) == "bitword(0b0, 1)"


def test_array():
    x = np.array([bitword(0b11), bitword(0b101)])
    assert x.dtype == bitword.dtype
    np.testing.assert_equal(x['val'], [0b11, 0b101])
    np.testing.assert_equal(x['bits'], [2, 3])


def test_array_repr():
    x = np.array([bitword(0b11), bitword(0b101)])
    assert repr(x) == "array([bitword( 0b11, 2), bitword(0b101, 3)])"


def test_zeros():
    x = np.zeros(3, dtype=bitword)
    assert x.dtype == bitword.dtype


def test_verify():
    x = np.zeros(2, dtype=bitword)
    x['val'][0] = 0b101
    x['bits'][0] = 2  # too small!
    with pytest.raises(ValueError) as exc_info:
        bitword.verify(x)
    exc_info.match(
        r"word lengths are inconsistent at indices:\n"
        r"\[0\]\n"
        r"with values:\n"
        r"\[bitword\(0b101, 2\)\]")
