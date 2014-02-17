#!/usr/bin/env python

from inspect import getmembers, ismethod

def update_docstrings(cls):
    for name, func in getmembers(cls):
        try:
            d1 = func.__doc__.split('\n')[0]
        except AttributeError:
            continue
        for parent in cls.__mro__[1:]:
            if '`%s`' % parent.__name__ in d1:
                d1 = d1.replace(parent.__name__, cls.__name__, 1)
                doc_ = func.__doc__.replace(parent.__name__, cls.__name__, 1)
                if ismethod(func):
                    func.__func__.__doc__ = doc_
                elif isinstance(func, property):
                    setattr(cls, name, property(doc=doc_, fget=func.fget,
                                                fset=func.fset, fdel=func.fdel))
    return cls
