from typing import Tuple


def rgb_to_hex(rgb: Tuple[int]) -> str:
    """
    Converts from RGB to hexadecimal representation of a color.

    Example
    -------
    >>> from polymesh.utils.colors import rgb_to_hex
    >>> rgb_to_hex((255, 255, 255))
    'ffffff'
    """
    return "%02x%02x%02x" % rgb


def hex_to_rgb(value: str) -> Tuple[int]:
    """
    Converts from hexadecimal to RGB representation of a color.

    Example
    -------
    >>> from polymesh.utils.colors import hex_to_rgb
    >>> hex_to_rgb("FF65BA")
    (255, 101, 186)
    """
    value = value.lstrip("#")
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))
