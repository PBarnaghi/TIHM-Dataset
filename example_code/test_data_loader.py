import numpy as np
import pytest
from data_loader import make_input_roll, TIHM, TIHMDataset
from sklearn import impute


def test_make_input_roll():
    # Test that the input roll works on a simple array
    assert np.all(
        make_input_roll(np.array([1, 2, 3, 4, 5]), 3)
        == np.array([[1, 2, 3], [2, 3, 4], [3, 4, 5]])
    )
    # test different errors
    with pytest.raises(TypeError):
        make_input_roll(np.array([1, 2, 3, 4, 5]), "3")
    with pytest.raises(TypeError):
        make_input_roll("not an array", 2)
    with pytest.raises(ValueError):
        make_input_roll(np.array([1, 2, 3, 4, 5]), 10)

    return


def test_TIHM():

    TIHM(root="./Dataset/")
    TIHM(root="./Dataset/")
    with pytest.raises(FileNotFoundError):
        TIHM(
            root="./some_where_that_does_not_contain_the_files/",
        )


def test_TIHMDataset():

    # data is loaded
    TIHMDataset(root="./Dataset/")
    TIHMDataset(root="./Dataset/")
    with pytest.raises(FileNotFoundError):
        TIHMDataset(
            root="./some_where_that_does_not_contain_the_files/",
        )
    TIHMDataset(root="./Dataset/", train=False)
    TIHMDataset(root="./Dataset/", train=True)

    # different normalisations
    TIHMDataset(root="./Dataset/", normalise=None)
    TIHMDataset(root="./Dataset/", normalise="global")
    TIHMDataset(root="./Dataset/", normalise="id")
    with pytest.raises(ValueError):
        TIHMDataset(root="./Dataset/", normalise="something_else")

    # testing different imputer
    TIHMDataset(root="./Dataset/", normalise="global", imputer=impute.KNNImputer())
    TIHMDataset(root="./Dataset/", normalise="global", imputer=impute.SimpleImputer())
    with pytest.raises(TypeError):
        TIHMDataset(root="./Dataset/", normalise="global", imputer=lambda x: x)
