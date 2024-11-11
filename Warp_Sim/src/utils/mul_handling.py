# mul_handling.py
import warp as wp
from typing import Union, TypeVar, Any

Number = TypeVar('Number', int, float, wp.float32, wp.float64)

class WarpNumber:
    """
    A wrapper class that handles operations between Python numbers and Warp types.
    Automatically converts results to wp.float64 when operating with Warp types.
    """
    def __init__(self, value: Number):
        self.value = value if isinstance(value, (wp.float32, wp.float64)) else wp.float64(value)
    
    @staticmethod
    def to_warp_float(value: Number) -> wp.float64:
        """Convert Python number types to warp float64"""
        if isinstance(value, WarpNumber):
            return value.value
        if isinstance(value, (wp.float32, wp.float64)):
            return value
        return wp.float64(value)

    def __mul__(self, other: Any) -> 'WarpNumber':
        if isinstance(other, (int, float, wp.float32, wp.float64, WarpNumber)):
            return WarpNumber(self.value * self.to_warp_float(other))
        return NotImplemented

    def __rmul__(self, other: Any) -> 'WarpNumber':
        return self.__mul__(other)
    
    def __add__(self, other: Any) -> 'WarpNumber':
        if isinstance(other, (int, float, wp.float32, wp.float64, WarpNumber)):
            return WarpNumber(self.value + self.to_warp_float(other))
        return NotImplemented

    def __radd__(self, other: Any) -> 'WarpNumber':
        return self.__add__(other)
    
    def __sub__(self, other: Any) -> 'WarpNumber':
        if isinstance(other, (int, float, wp.float32, wp.float64, WarpNumber)):
            return WarpNumber(self.value - self.to_warp_float(other))
        return NotImplemented

    def __rsub__(self, other: Any) -> 'WarpNumber':
        if isinstance(other, (int, float, wp.float32, wp.float64, WarpNumber)):
            return WarpNumber(self.to_warp_float(other) - self.value)
        return NotImplemented
    
    def __truediv__(self, other: Any) -> 'WarpNumber':
        if isinstance(other, (int, float, wp.float32, wp.float64, WarpNumber)):
            return WarpNumber(self.value / self.to_warp_float(other))
        return NotImplemented

    def __rtruediv__(self, other: Any) -> 'WarpNumber':
        if isinstance(other, (int, float, wp.float32, wp.float64, WarpNumber)):
            return WarpNumber(self.to_warp_float(other) / self.value)
        return NotImplemented
    
    def __neg__(self) -> 'WarpNumber':
        return WarpNumber(-self.value)
    
    def __float__(self) -> float:
        return float(self.value)
    
    def __repr__(self) -> str:
        return f"WarpNumber({self.value})"

# Convenience function to wrap numbers
def warp(x: Number) -> WarpNumber:
    """Convenience function to create WarpNumber instances"""
    return WarpNumber(x)

# Test function to verify operations
def test_operations():
    """Test various operations between Python numbers and Warp types"""
    # Create test values
    w1 = warp(2.0)
    w2 = warp(3.0)
    f1 = 2.0
    wf = wp.float64(3.0)
    
    # Test multiplication
    assert isinstance((w1 * f1).value, wp.float64)
    assert isinstance((f1 * w1).value, wp.float64)
    assert isinstance((w1 * wf).value, wp.float64)
    
    # Test addition
    assert isinstance((w1 + f1).value, wp.float64)
    assert isinstance((f1 + w1).value, wp.float64)
    assert isinstance((w1 + wf).value, wp.float64)
    
    # Test subtraction
    assert isinstance((w1 - f1).value, wp.float64)
    assert isinstance((f1 - w1).value, wp.float64)
    assert isinstance((w1 - wf).value, wp.float64)
    
    # Test division
    assert isinstance((w1 / f1).value, wp.float64)
    assert isinstance((f1 / w1).value, wp.float64)
    assert isinstance((w1 / wf).value, wp.float64)
    
    print("All operations tests passed!")

if __name__ == "__main__":
    test_operations()