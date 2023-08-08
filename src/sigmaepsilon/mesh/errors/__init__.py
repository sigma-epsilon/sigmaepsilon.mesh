from numpy import ndarray

from sigmaepsilon.core.misc import get_index_suffix
from sigmaepsilon.core.exceptions import SigmaEpsilonException


class ArrayShapeMismatchError(SigmaEpsilonException):
    """
    An exception for the case for when one or more axes of an array does not
    have the desired length.
    """

    def __init__(
        self,
        message: str = None,
        *,
        arr_name: str = None,
        arr: ndarray = None,
        index: int = None,
        expected: int = None,
    ):
        if message is None:
            assert (
                arr is not None
                and arr_name is not None
                and index is not None
                and expected is not None
            )
            suffix = get_index_suffix(index)
            message = f"The {index}{suffix} axis of {arr_name} is {arr.shape[index]} but {expected} is expected."
        self.message = message
        super().__init__(message)

    @classmethod
    def check_axis_length(cls, arr: ndarray, arr_name: str, index: int, expected: int):
        if arr.shape[index] != expected:
            raise ArrayShapeMismatchError(
                arr=arr, arr_name=arr_name, index=index, expected=expected
            )
