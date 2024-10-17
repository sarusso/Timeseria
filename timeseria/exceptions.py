# -*- coding: utf-8 -*-
"""Exceptions."""


class ConsistencyException(Exception):
    """Rasied when the internal consistency is broken."""
    pass

class NotFittedError(Exception):
    """Raised when trying to save, apply or evaluate a model that requires fitting first."""
    pass

class AlreadyFittedError(Exception):
    """Raised when trying to fit a model that is already fitted (instead of using the fit update method, if available)."""
    pass

class NonContiguityError(Exception):
    """Raised when the model only supports being applied on data contiguous with the fit data and it is not."""
    pass

class NoDataException(Exception):
    """Raised if a storage get() function finds no  data at all."""
    pass

class FloatConversionError(Exception):
    """Raised to group the various exceptions that can lead to the impossibility of converting a value to a floating point."""
    pass

class NotEnoughDataError(Exception):
    """Raised in context when there is not enough data to perform the required operation"""



