# -*- coding: utf-8 -*-
"""Exceptions."""


class ConsistencyException(Exception):
    """Rasied when the internal consistency is broken."""
    pass

class NotFittedError(Exception):
    """Raised when trying to save, apply or evaluate a model that requires fitting first."""
    pass

class NonContiguityError(Exception):
    """Raised when the model only supports being applied on data contiguous with the fit data and it is not."""
    pass

class NoDataException(Exception):
    """Raised if a storage get() function finds no  data at all."""
    pass

class FloatConversionError(Exception):
    """Raise to group the various exceptions that can lead to the impossibility of converting a value to a floating point."""
    pass
