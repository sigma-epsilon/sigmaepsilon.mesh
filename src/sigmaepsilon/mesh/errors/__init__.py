from numpy import ndarray


class ArrayShapeMismatchError(Exception):
    """
    An exception for the case when an axis of an array does not
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
            suffix = self.get_index_suffix(index)
            message = f"The {index}{suffix} axis of {arr_name} is {arr.shape[index]} but {expected} is expected."
        self.message = message
        super().__init__(message)

    def get_index_suffix(self, index: int) -> str:
        if index % 100 in {11, 12, 13}:  # Special case for 11th, 12th, and 13th
            suffix = "th"
        else:
            last_digit = index % 10
            if last_digit == 1:
                suffix = "st"
            elif last_digit == 2:
                suffix = "nd"
            elif last_digit == 3:
                suffix = "rd"
            else:
                suffix = "th"
        return suffix

    @classmethod
    def check_axis_length(cls, arr: ndarray, arr_name: str, index: int, expected: int):
        if arr.shape[index] != expected:
            raise ArrayShapeMismatchError(
                arr=arr, arr_name=arr_name, index=index, expected=expected
            )
