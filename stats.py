#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Some math utilities

# Copyright (C) 2025-2026  Marco Chieppa | crap0101

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not see <http://www.gnu.org/licenses/>

from collections import Counter, defaultdict
from collections.abc import Callable, ItemsView, Sequence
from enum import Enum
import math
import operator
import random
from typing import Any, Generic

#
import inspect
# local imports
import rand

#############
# some defs #
#############

Number = float | int

class CumFreqT(Enum):
    lt = 0
    gt = 1

class StrCls(type):
    """Pretty print class names."""
    def __str__(self):
        return self.name
    __repr__ = __str__

class GDTYPE:
    """A class for managing grouped data types."""
    __metaclass__ = Generic

    class basic(metaclass=StrCls):
        name = 'basic'
        mean = ... # ugmean (later, undef her)
        data2freq = ... # make_data_freq

    class freq(metaclass=StrCls):
        name = 'freq'
        mean = ... # fgmean
        data2freq = ... # lambda x: x

    class interval(metaclass=StrCls):
        name = 'interval'
        mean = ... # igmean
        data2freq = ... # freqs_from_intervals

    def __class_getitem__(cls, item):
        if item in ('basic', cls.basic):
            return cls.basic
        elif item in ('freq', cls.freq):
            return cls.freq
        elif item in ('interval', cls.interval):
            return cls.interval
        else:
            raise TypeError(f"No such item '{item}'")


class IntervalError(Exception):
    """Base error class for IntevalDict objects."""
    def __init__ (self, msg):
        self.msg = msg
        super().__init__(msg)
    def __str__ (self):
        return self.msg

# ...also see below for GDTYPE

###############
# general use #
###############

def get_pop (size: Number,
             minval: Number,
             maxval: Number) -> Sequence[Number, ...]:
    """Returns a random list of $size elements between $minval and $maxval (inclusive)."""
    #return [random.randrange(minval, maxval+1) for _ in range(size)]
    return list(rand.getrand(minval, maxval+1, size))

def get_sample (population: Sequence[Number, ...],
                size: Number) -> Sequence[Number, ...]:
    """Returns a random sample of $size elements from $population."""
    return random.choices(population, k=size)


######################
# for ungrouped data #
######################

def deviation (mean: Number, score: Number) -> Number:
    """Deviation of $score from the $mean."""
    return score - mean

def mean (group: Sequence[Number, ...]) -> Number:
    """$group's mean for ungrouped data."""
    return sum(group) / len(group)

def median (data: Sequence):
    """
    Returns the median of $data for ungrouped data, assuming data is a sorted sequence
    supporting the __len__ and __getitem__ methods.
    >>> median((24, 34, 43, 50, 67, 78))
    46.5
    >>> median((23, 34, 43, 54, 56, 67, 78))
    54
    """
    n = len(data)
    if n & 1:
        return data[int(n / 2)]
    return (data[int(n / 2) - 1] + data[int(n / 2)]) / 2

def _variance (mean: Number,
               data: Sequence[Number, ...],
               fromsample: bool) -> Number:
    """Variance for ungrouped data."""
    return sum(deviation(mean, score)**2 for score in data) / (len(data) - (1 if fromsample else 0))

def variance (data: Sequence[Number, ...],
              fromsample: bool = False) -> Number:
    """
    Variance of $data for ungrouped data.
    Set $fromsample to True if $data is a sample.
    """
    return _variance(mean(data), data, fromsample)

def _standard_dev (mean, data, fromsample):
    return math.sqrt(_variance(mean, data, fromsample))

def standard_dev (data: Sequence[Number, ...],
                  fromsample: bool = False) -> Number:
    """
    Standard deviation (using the Actual Mean Method) for ungrouped data.
    Set $fromsample to True if $data is a sample.
    """
    return _standard_dev(mean(data), data, fromsample)

def _standard_error (mean, data, fromsample):
    """Standard error for ungrouped data."""
    return _standard_dev(mean, data, fromsample) / math.sqrt(len(data))

def standard_error (data: Sequence[Number, ...],
                    fromsample: bool = False) -> Number:
    """
    Standard error (using the Actual Mean Method) for ungrouped data.
    Set $fromsample to True if $data is a sample.
    """
    return _standard_error(mean(data), data, fromsample)


####################
# for grouped data #
###################

def autogroup_perc_chunks (data_length: Number, perc: Number = 20) -> Number:
    """
    Return the number of chunks to be used with the autogroup function based on
    the $data_length to make sure that each chunks contains at least $perc % values.
    >>> autogroup_perc_chunks(50)
    5
    >>> autogroup_perc_chunks(100)
    5
    >>> autogroup_perc_chunks(100, 50)
    2
    >>> autogroup_perc_chunks(100, 60)
    1
    >>> autogroup_perc_chunks(100, 1)
    100
    >>> autogroup_perc_chunks(100, 30)
    3
    """
    if perc < 1 or perc > 100:
        raise ValueError(f"wrong percentage value: {perc}")
    chunks = int(data_length / (data_length * (perc / 100)))
    if chunks < 1:
        raise ValueError("not enough data for make chunks")
    return chunks

def autogroup (chunks: Number,
               minvalue: Number,
               maxvalue: Number,
               overlap: bool = False) -> Sequence[Sequence[Number,Number], ...]:
    """
    Returns class intervals in N $chunks from $minvalue to $maxvalue.
    $overlap (default: False) can be used for build overlapping intervals, e.g.
        >>> autogroup(5,0,50)
        [(0, 10), (11, 21), (22, 32), (33, 43), (44, 50)]
        >>> autogroup(5,0,50,True)
       [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50)]
    """
    if minvalue >= maxvalue:
        raise ValueError(f"minvalue >= maxvalue: ({minvalue} >= {maxvalue})")
    if (maxvalue - minvalue) < chunks:
        raise ValueError("not enough values for make intervals")
    q = maxvalue - minvalue
    nc = int(q / chunks)
    if nc < 1:
        raise ValueError(f"can't make chunks from this data length")
    overlap = 0 if overlap else 1
    classes = []
    while minvalue < maxvalue:
        if minvalue + nc >= maxvalue:
            classes.append((minvalue, maxvalue))
            break
        else:
            classes.append((minvalue, minvalue + nc))
            minvalue += nc + overlap
    return classes


class IntervalDict:
    """
    Object for class intervals.
    """
    def __init__ (self,
                  intervals: Sequence[Sequence[Number,Number], ...] = (),
                  data: None|Sequence[Number, ...] = None,
                  trim : bool = False,
                  overlap: bool = False):
        """
        $intervals is a sequence of (min, max) pairs, filled with the optional values from $data.
        If $trim is True, ignores values from $data which don't fit in the interval, otherwise raises a IntervalError.
        $overlap (default: False) has the same meaning as in autogroup(), so set it to True
        when using overlapping intervals.
        >>> i = IntervalDict([(0,10),(10,20)], range(1,40,3))
        Traceback (most recent call last)
        ...
        IntervalError: value "22" doesn't belong to any interval
        >>> i = IntervalDict([(0,10),(10,20)], range(1,40,3), trim=True)
        >>> i
        {(0, 10): [1, 4, 7, 10], (10, 20): [13, 16, 19]}
        >>> i
        {(0, 10): [1, 4, 7, 10, 10], (10, 20): [13, 16, 19]}
        >>> i = IntervalDict([(0,10),(10,20)], range(1,40,3), trim=True, overlap=True)
        >>> i
        {(0, 10): [1, 4, 7], (10, 20): [10, 13, 16, 19]}
        >>> i +=10
        >>> i
        {(0, 10): [1, 4, 7], (10, 20): [10, 13, 16, 19, 10]}
        """
        self._dict = {tuple(i):[] for i in intervals}
        self._overlap = bool(overlap)
        if self._overlap is True:
            self._overlap_cmp = operator.lt
        else:
            self._overlap_cmp = operator.le
        if data:
            self.extend(data, not bool(trim))     

    def __delitem__ (self, value: Number) -> None:
        """
        Deletes the class interval (and its values) in which $value *may* belong or raises IntervalError.
        >>> i = IntervalDict(((0,11),(11,30),(31,40)), range(5,40,4))
        >>> i
        {(0, 11): [5, 9], (11, 30): [13, 17, 21, 25, 29], (31, 40): [33, 37]}
        >>> del i[8] # 8 is not in the (0, 11) interval, but can be. Use remove_interval_by_value for removing
        >>> ...      #  the interval for existing values only. 
        >>> i
        {(11, 30): [13, 17, 21, 25, 29], (31, 40): [33, 37]}
        >>> del i[13]
        >>> i
        {(31, 40): [33, 37]}
        >>> del i[213]
        Traceback (most recent call last):
        ...
        IntervalError: value '213' not in intervals
        """
        for (imin, imax), _ in self._dict.items():
            if value >= imin and self._overlap_cmp(value, imax):
                break
        else:
            raise IntervalError(f"value '{value}' not in intervals")
        del self._dict[(imin, imax)]

    def __getitem__ (self, value: Number) -> Sequence[Sequence[Number,Number], Sequence[Number,...]]:
        """
        Returns the interval's class and values in which $value *may* belong or raises IntervalError.
        >>> i = IntervalDict(((0,11),(11,30),(31,40)), range(0,40,4))
        >>> i
        {(0, 11): [0, 4, 8], (11, 30): [12, 16, 20, 24, 28], (31, 40): [32, 36]}
        >>> i[7] # 7 is not in the interval's values, but can be. Use get_interval to get the interval for existing values only.
        ((0, 11), (0, 4, 8))
        >>> i[16]
        ((11, 30), (12, 16, 20, 24, 28))
        >>> i[99]
        Traceback (most recent call last):
        ...
        IntervalError: value '99' not in intervals
        """
        for (imin, imax), seq in self._dict.items():
            if value >= imin and self._overlap_cmp(value, imax):
                return (imin, imax), tuple(seq)
        raise IntervalError(f"value '{value}' not in intervals")

    def __iadd__ (self, value: Number):
        """
        Adds $value to it's belonging interval or raises IntervalError.
        >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 4))
        >>> i
        {(0, 10): [0, 4, 8], (11, 20): [12, 16]}
        >>> for n in range(5): i += n
        ... 
        >>> i
        {(0, 10): [0, 4, 8, 0, 1, 2, 3, 4], (11, 20): [12, 16]}
        >>> i += 99
        Traceback (most recent call last):
        ...
        IntervalError: value "99" doesn't belong to any interval
        """
        for (imin, imax), _ in self._dict.items():
            if value >= imin and self._overlap_cmp(value, imax):
                self._dict[(imin, imax)].append(value)
                return self
        raise IntervalError(f'''value "{value}" doesn't belong to any interval''')

    def __iter__ (self):
        """NotImplemented"""
        raise NotImplementedError(f"{self.__class__.__name__} {inspect.getframeinfo(inspect.currentframe()).function}")

    def __len__ (self) -> Number:
        """Returns the number of intervals."""
        return len(self._dict)

    def __repr__ (self):
        return repr(self._dict)
    __str__ = __repr__

    def add_interval (self,
                      interval: Sequence[Number, Number],
                      values: Sequence[Number, ...]) -> None:
        """
        Adds an $interval and the corresponding $values.
        NOTE: this causes the rebuild of the underlying dictionary.
        Raises IntervalError is $interval is already present.
        >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 3))
        >>> i
        {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
        >>> i.add_interval((0,10), range(9))
        Traceback (most recent call last):
        ....
        IntervalError: interval (0, 10) already present
        >>> i.add_interval((10,20), range(9))
        >>> i
        {(0, 10): [0, 3, 6, 9], (10, 20): [0, 1, 2, 3, 4, 5, 6, 7, 8], (11, 20): [12, 15, 18]}
        >>> # Note as at this time there aren't checks for inconsistent data, left to the user's responsibility
        """
        if tuple(interval) in self._dict.keys():
            raise IntervalError(f'interval {interval} already present')
        self._dict[tuple(interval)] = list(values)
        self._dict = dict(sorted(self._dict.items())) #XXX+TODO: to sort when get smt only?

    def empty_interval (self, interval: Sequence[Number,Number]):
        """
        Removes all the values from $Interval. Raises IntervalError if $interval doesn't exist.
        >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 3))
        >>> i
        {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
        >>> i.empty_interval((0,5))
        Traceback (most recent call last):
        ...
        IntervalError: Interval (0, 5) not found
        >>> i
        {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
        >>> i.empty_interval((0,10))
        >>> i
        {(0, 10): [], (11, 20): [12, 15, 18]}
        """
        target = tuple(interval)
        if target not in self._dict.keys():
            raise IntervalError(f"Interval {interval} not found")
        self._dict[target] = []

    def extend (self,
                seq: Sequence[Number, ...],
                err: bool = True) -> None:
        """
        Adds $seq's values to their belonging intervals.
        If $err is True (the default) raises a IntervalError for values which don't fit in any interval;
        in this case the values ​​just entered will be removed, otherwise only the valid values
        ​​will be entered, ignoring the incorrect ones.
        >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 4))
        >>> i
        {(0, 10): [0, 4, 8], (11, 20): [12, 16]}
        >>> i.extend(range(5))
        >>> i
        {(0, 10): [0, 4, 8, 0, 1, 2, 3, 4], (11, 20): [12, 16]}
        >>> i.extend(range(15,25))
        Traceback (most recent call last):
        ...
        IntervalError: value "21" doesn't belong to any interval
        >>> i
        {(0, 10): [0, 4, 8, 0, 1, 2, 3, 4], (11, 20): [12, 16]}
        >>> i.extend(range(15,25), err=False)
        >>> i
        {(0, 10): [0, 4, 8, 0, 1, 2, 3, 4], (11, 20): [12, 16, 15, 16, 17, 18, 19, 20]
        """
        todel = []
        for item in seq:
            try:
                self += item
                todel.append(item)
            except IntervalError as e:
                if err:
                    for value in todel:
                        self.remove_value(value)
                    raise e

    def get_interval (self, value: Number) -> Sequence[Number,Number]:
        """
        Returns the class interval in which $value belong or raises IntervalError.
        >>> i = IntervalDict(((0,11),(11,30),(31,40)), range(5,40,4))
        >>> i
        {(0, 11): [5, 9], (11, 30): [13, 17, 21, 25, 29], (31, 40): [33, 37]}
        >>> i.get_interval(17)
        (11, 30)
        >>> i.get_interval(7)
        Traceback (most recent call last):
        ...
        IntervalError: value '7' not in interval
        """
        for cls, seq in self._dict.items():
            if value in seq:
                return cls
        raise IntervalError(f"value '{value}' not in intervals")

    def get_values (self, cls: Sequence[Number,Number]) -> Sequence[Number,...]:
        """
        Returns the values belonging to the class interval $cls or raises IntervalError.
        >>> i = IntervalDict(((0,10),(11,30)), range(5,40,4), True)
        >>> i
        {(0, 10): [5, 9], (11, 30): [13, 17, 21, 25, 29]}
        >>> i.get_values((0,9))
        Traceback (most recent call last):
        ...
        IntervalError: class '(0, 9)' not found
        >>> i.get_interval(5)
        (0, 10)
        >>> i.get_values(i.get_interval(5))
        (5, 9)
        """
        target = tuple(cls)
        for interval, seq in self._dict.items():
            if tuple(interval) == target:
                return tuple(seq)
        raise IntervalError(f"class '{cls}' not found")

    @property
    def intervals (self) -> Sequence[Sequence[Number,Number], ...]:
        """The class intervals."""
        return tuple(self._dict.keys())

    def items (self) -> ItemsView:
        """Returns a new view of the underling dictionary's items."""
        return self._dict.items()

    @property
    def length (self) -> Number:
        """The number of values in all the intervals."""
        return sum(len(v) for v in self._dict.values())

    @property
    def overlap (self) -> bool:
        """A bool, if the IntervalDict is an overlapping ones or not."""
        return self._overlap

    def remove_interval (self, interval: Sequence[Number,Number]) -> None:
        """
        Removes $interval and its values or raises IntervalError.
        >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 3))
        >>> i
        {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
        >>> i.remove_interval((11,21))
        Traceback (most recent call last):
        ...
        IntervalError: interval (11, 21) not found
        >>> i
        {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
        >>> i.remove_interval([11,20])
        >>> i
        {(0, 10): [0, 3, 6, 9]}
        """
        target = tuple(interval)
        for cls in self._dict.keys():
            if tuple(cls) == target:
                del self._dict[cls]
                return
        raise IntervalError(f"interval {interval} not found")
            
    def remove_interval_by_value (self, value: Number) -> None:
        """
        Removes the (first, obviously) interval in which $value belong or raises IntervalError.
        >>> i = IntervalDict(((0,11),(11,30),(31,40)), range(5,40,4))
        >>> i
        {(0, 11): [5, 9], (11, 30): [13, 17, 21, 25, 29], (31, 40): [33, 37]}
        >>> i.remove_interval_by_value(8)
        Traceback (most recent call last):
        ...
        IntervalError: value 8 not in intervals
        >>> i
        {(0, 11): [5, 9], (11, 30): [13, 17, 21, 25, 29], (31, 40): [33, 37]}
        >>> i.remove_interval_by_value(9)
        >>> i
        {(11, 30): [13, 17, 21, 25, 29], (31, 40): [33, 37]}
        """
        for cls, seq in self._dict.items():
            if value in seq:
                del self._dict[cls]
                return
        raise IntervalError(f"value {value} not in intervals")
        
    def remove_value (self, value: Number,
                      all: bool = False) -> None:
        """
        Removes the first occurrence of $value from its interval.
        If $all is True, remove all the occurrences.
        Raises IntervalError if $value is not present.
        >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 3))
        >>> i.remove_value(5)
        Traceback (most recent call last):
        ...
        IntervalError: value '5' not in intervals
        >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 3))
        >>> i
        {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
        >>> for _ in range(3): i += 4
        ... 
        >>> i
        {(0, 10): [0, 3, 6, 9, 4, 4, 4], (11, 20): [12, 15, 18]}
        >>> i.remove_value(4)
        >>> i
        {(0, 10): [0, 3, 6, 9, 4, 4], (11, 20): [12, 15, 18]}
        >>> i.remove_value(4, all=True)
        >>> i
        {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
        >>> i.remove_value(4)
        Traceback (most recent call last):
        ...
        IntervalError: value '4' not in intervals
        """
        for (imin, imax), seq in self._dict.items():
            if value >= imin and value <= imax:
                try:
                    seq.remove(value)
                except ValueError as e:
                    raise IntervalError(f"value '{value}' not in intervals") from None
                if all:
                    while value in seq:
                        seq.remove(value)
                return
        raise IntervalError(f"value '{value}' not in intervals")


def cumulative_freq (data: Sequence,
                     limit: Number|Sequence[Number,Number], # but can works with appropriates Any
                     ftype:CumFreqT = CumFreqT.lt,
                     cmpfunc: Callable[[Any,Any], Sequence[bool,Number]] = lambda x,y:(x<=y,x)) -> Number:
    """
    Returns the cumulative frequency for $data (a sequence supporting reversed()).
    $data is assumed to be ordered.
    $limit is the boundary's class interval for the cumulative count.
    $ftype if the cumulative type:
        CumFreqT.lt for the "lesser than" cumulative frequency (default), or
        CumFreqT.gt for the "greater than" cumulative frequency
    $cmpfunc can be any function accepting two parameter (1st, 2nd) where:
        1st: is the item of the sequence
        2nd: is $limit
    and returning a pair of (a, b) where:
        a: (bool) indicating succerful comparison
        b: a value which will be added to the cumulative frequency
    $cmpfunc by default returns (1st <= 2nd, 1st).
    Note: limit is mandatory, since getting the cumulative freqs for all the sequence
          is even simpler and can be done with specific code for the particular case.
          Anyway, the total cumulative freq can be computed, for example, as:
          >>> d
          [(1, 10), (2, 21), (3, 20), (4, 26), (5, 20), (6, 4)]
          >>> cumulative_freq(d,None,cmpfunc=lambda x,y:(True, x[1]))
          101
    >>> data = list(range(10))
    >>> data
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    >>> cumulative_freq(data, 5)
    10
    >>> cumulative_freq(data, 5, CumFreqT.gt, lambda x,y:(x>y,x))
    30
    >>> i = IntervalDict([(0,5),(5,10),(10,15)], range(12))
    >>> i
    {(0, 5): [0, 1, 2, 3, 4, 5], (5, 10): [6, 7, 8, 9, 10], (10, 15): [11]}
    >>> cumulative_freq(i.items(), (5,10), cmpfunc=lambda x,y:(x[0]<y,len(x[1])))
    6
    >>> cumulative_freq(i.items(), (5,10), cmpfunc=lambda x,y:(x[0]<=y,len(x[1]))) # (5, 10) inclusive
    11
    >>> cumulative_freq(i.items(), None, cmpfunc=lambda x,y:(True,len(x[1]))) # all
    12
    >>> d = sorted(make_data_freq([0,2,2,0,4,4,10,10,10,1,1,1,1,0,10,10,1,1]))
    >>> d
    [(0, 3), (1, 6), (2, 2), (4, 2), (10, 5)]
    >>> cumulative_freq(d, 4, cmpfunc=lambda x,y:(x[0]<y,x[1])) # until (4,2) excluded
    11
    >>> cumulative_freq(d, 4, cmpfunc=lambda x,y:(x[0]<=y,x[1])) # until (4,2) included
    13
    >>> cumulative_freq(d, (4,2), cmpfunc=lambda x,y:(x<=y,x[1])) # using directly the item for comparison
    13
    >>> cumulative_freq(d, 4, CumFreqT.gt, cmpfunc=lambda x,y:(x[0]>y,x[1]))
    5
    >>> d = [(1, 10), (2, 21), (3, 20), (4, 26), (5, 20), (6, 4)]
    >>> cumulative_freq(d,3,cmpfunc=lambda x,y:(x[0]<y,x[1]))
    31
    >>> cumulative_freq(d,3,cmpfunc=lambda x,y:(x[0]<=y,x[1]))
    51
    """ 
    freq = 0
    if not isinstance(ftype, CumFreqT):
        raise ValueError(f"wrong value for ftype: {ftype}")
    if ftype is CumFreqT.gt:
        data = reversed(data)
    for item in data:
        ok, value = cmpfunc(item, limit)
        if ok:
            freq += value
        else:
            break
    return freq

def freqs_from_intervals (data: IntervalDict) -> Sequence[Sequence[Number,Number], ...]:
    """
    Return frequencies from interval data.
    >>> make_data_intervals([0,1,2], [(0,5),(6,10)])
    {(0, 5): [0, 1, 2], (6, 10): []}
    >>> freqs_from_intervals(make_data_intervals([0,1,2], [(0,5),(6,10)]))
    [(2.5, 3), (8.0, 0)]
    """
    freqs = []
    for (imin, imax), values in data.items():
        freqs.append((((imax + imin) / 2 ), len(values)))
    return freqs

def freqs_get_class (data: Sequence[Sequence[Number,Number], ...],
                     cls: Any) -> Sequence[Any, Any]:
    """
    Returns the (class, freq) pair belonging to $cls from the the frequencies sequence $data.
    Raise ValueError when not found.
    >>> dm
    [(0, 1), (1, 6), (2, 7), (3, 2), (4, 3), (5, 1)]
    >>> freqs_get_class(dm, 3)
    (3, 2)
    >>> freqs_get_class(dm, 13)
    Traceback (most recent call last):
    ...
    ValueError: class 13 not found
    """
    for c, f in data:
        if c == cls:
            return (c, f)
    raise ValueError(f"class {cls} not found")

def make_data_intervals (data: Sequence[Number, ...],
                         intervals: Sequence[Sequence[Number,Number], ...],
                         trim=False,
                         overlap=False) -> IntervalDict:
    """
    Returns an IntervalDict with the given $intervals fillen with values from $data.
    $trim and $overlap has the same meaning as in in the IntervalDict constructor.
    >>> make_data_intervals([0,1,2], [(0,5),(6,10)])
    {(0, 5): [0, 1, 2], (6, 10): []}
    """
    d = IntervalDict(intervals, trim=trim, overlap=overlap)
    for value in data:
        d += value
    return d

def make_data_intervals_from_freq (data: Sequence[Sequence[Number,Number], ...],
                                   size: Number,
                                   trim=False,
                                   overlap=False) -> IntervalDict:
    """
    Returns an IntervalDict from $data frequencies with class intervals of $size.
    $data is a sequence of pair in the the form of (lower_class_value, frequency_value), so
    a class intervals will be, e.g. (lower_class_value, lower_class_value + $size),
    while its values ​​are the class midpoint (as many as frequency_value).
    $trim and $overlap has the same meaning as in in the IntervalDict constructor.
    NOTE: this is mainly a convenience function for some operations
    (fake values can be changed afterwards, anyway).
    >>> make_data_intervals_from_freq([(0,3),(5,2),(10,5)], 5, overlap=True)
    {(0, 5): [2.5, 2.5, 2.5], (5, 10): [7.5, 7.5], (10, 15): [12.5, 12.5, 12.5, 12.5, 12.5]}
    """
    d = IntervalDict(trim=trim, overlap=overlap)
    for class_lower, amount in data:
        class_upper = class_lower + size
        d.add_interval([class_lower, class_upper], [(class_lower + class_upper) / 2] * amount)
    return d

def make_data_freq (data: Sequence[Number]) -> Sequence[[Number,Number]]:
    """
    Returns frequencies from distinct measurements.
    >>> from itertools import chain
    >>> make_data_freq(chain(*[range(5), range(1,10,2)]))
    [(0, 1), (1, 2), (2, 1), (3, 2), (4, 1), (5, 1), (7, 1), (9, 1)]
    """
    freqd = defaultdict(int)
    for value in data:
        freqd[value] += 1
    return list(freqd.items())

def _median_class_and_cumfreq (data: Sequence[Sequence[Any,Any], ...],
                              percentile: Number = 50, vfunc=lambda x:x) -> Sequence[Any,Number]:
    """
    Returns the median class and the relative cumulative frequence of $data at the target $percentile.
    $data is a sequence of (class, values_or_freq) pairs, while $vfunc is a function to be applied to
    values_or_freq before the computation of the cumulative frequence (default to the identity function).
    NOTE: test: not considering if observations are even or odd.
    """
    if (percentile <= 0) or (percentile > 100):
        raise ValueError(f"wrong percentile value: {percentile}")
    target_value = sum(vfunc(v) for _, v in data) * percentile / 100
    cumf = 0
    for cls, values in data:
        cumf += vfunc(values)
        if cumf >= target_value:
            return cls, cumf
    raise ValueError("median class not found")

def median_class_and_cumfreq (data: Sequence[Sequence[Any,Any], ...],
                              percentile: Number = 50, vfunc=lambda x:x) -> Sequence[Any,Number]:
    """
    Returns the median class and the relative cumulative frequence of $data at the target $percentile.
    $data is a sequence of (class, values_or_freq) pairs, while $vfunc is a function to be applied to
    values_or_freq before the computation of the cumulative frequence (default to the identity function).
    >>> dm
    [(0, 1), (1, 6), (2, 7), (3, 2), (4, 3), (5, 1)]
    >>> median_class_and_cumfreq(dm)
    (2, 14)
    >>> i = IntervalDict([(0,5),(5,10),(10,15)], [1,1,2,4,5,6,9,10,11,13], overlap=True)
    >>> i
    {(0, 5): [1, 1, 2, 4], (5, 10): [5, 6, 9], (10, 15): [10, 11, 13]}
    >>> median_class_and_cumfreq(i.items(), vfunc=lambda x:len(x))
    ((5, 10), 7)
    """
    if (percentile <= 0) or (percentile > 100):
        raise ValueError(f"wrong percentile value: {percentile}")
    tot = sum(vfunc(v) for _, v in data)
    if tot & 1:
        target_value = tot * percentile / 100
    else:
        target_value = ((tot * percentile / 100) + (1 + (tot * percentile / 100))) / 2
    cumf = 0
    for cls, values in data:
        cumf += vfunc(values)
        if cumf >= target_value:
            return cls, cumf
    raise ValueError("median class not found")

def median_class_and_cumfreq_at_value (data: Sequence[Sequence[Any,Any], ...],
                              target_freq: Number, vfunc=lambda x:x) -> Sequence[Any,Number]:
    """
    Returns the median class and the relative cumulative frequence of $data at the $target_freq point.
    $data is a sequence of (class, values_or_freq) pairs, while $vfunc is a function to be applied to
    values_or_freq before the computation of the cumulative frequence (default to the identity function).
    >>> dm
    [(0, 1), (1, 6), (2, 7), (3, 2), (4, 3), (5, 1)]
    >>> median_class_and_cumfreq(dm)
    (2, 14)
    >>> median_class_and_cumfreq_at_value(dm, 14)
    (2, 14)
    >>> median_class_and_cumfreq_at_value(dm, 2)
    (1, 7)
    >>> median_class_and_cumfreq_at_value(dm, 19)
    (4, 19)
    """
    cumf = 0
    for cls, values in data:
        cumf += vfunc(values)
        if cumf >= target_freq:
            return cls, cumf
    raise ValueError("median class (at value) not found")

# means:

def ugmean (data: Sequence[Number]) -> Number:
    """
    $data's mean for grouped data without class intervals (using the Direct Method).
    >>> ugmean([1,2,3])
    2.0
    >>> ugmean([1,2,3,10])
    4.0
    """
    tot = len(data)
    return fgmean(make_data_freq(data))
    #freqd = make_data_freq(data)
    #return sum(i * f for i, f in freqd.items()) / sum(freqd.values())

def fgmean (data: Sequence[Sequence[Number,Number], ...]) -> Number:
    """
    $data's mean for grouped data in (value, freq) format (using the Direct Method).
    >>> i
    {(0, 5): [0, 1, 2], (6, 10): []}
    >>> fgmean(freqs_from_intervals(i))
    2.5
    """
    tval = 0
    fval = 0
    for value, freq in data:
        tval += value * freq
        fval += freq
    return tval / fval

def igmean (data: IntervalDict) -> Number:
    """
    $data's mean for grouped data with class intervals (using the Direct Method).
    {(0, 5): [0, 1, 2], (6, 10): []}
    >>> igmean(i)
    2.5
    """
    total_class = 0
    total_freq = 0
    for (imin, imax), values in data.items():
        lv = len(values)
        total_class += ((imax + imin) / 2 ) * lv
        total_freq += lv
    return total_class / total_freq

def gmean (data, dtype=None) -> Number:
    """
    Try guessing data type for the correct mean func, or raise TypeError.
    $data can be an IntervalDict, a sequence of (value, freq) pairs o a sequence of values.
    >>> data = list(sorted(chain(*[range(20),range(1,20,3), range(1,20,2),[0,1,2]*5])))
    >>> data
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6,
     7, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13, 13, 14, 15, 15, 16, 16, 17, 17, 18, 19, 19, 19]
    >>> i = make_data_intervals(data, [(0,5),(6,10),(10,15),(16,20)])
    >>> i
    {(0, 5): [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5],
     (6, 10): [6, 7, 7, 7, 8, 9, 9, 10, 10],
     (10, 15): [11, 11, 12, 13, 13, 13, 14, 15, 15],
     (16, 20): [16, 16, 17, 17, 18, 19, 19, 19]}
    >>> gmean(i) # igmean
    7.5673076923076925
    >>> gmean(freqs_from_intervals(i)) # fgmean
    7.5673076923076925
    >>> gmean(list(chain(*(v for k,v in i.items())))) # ugmean
    7.211538461538462
    """
    if dtype is None:
        if isinstance(data, IntervalDict):
            dtype = GDTYPE.interval
        elif isinstance(data, Sequence):
            if isinstance(data[0], Number):
                dtype = GDTYPE.basic
            elif isinstance(data[0], Sequence):
                dtype = GDTYPE.freq
    return GDTYPE[dtype].mean(data)

GDTYPE.basic.mean = ugmean
GDTYPE.basic.data2freq = make_data_freq
GDTYPE.freq.mean = fgmean
GDTYPE.freq.data2freq = lambda x: x
GDTYPE.interval.mean = igmean
GDTYPE.interval.data2freq = freqs_from_intervals

# median:

def fmedian (data: Sequence[Sequence[Number,Number], ...]) -> Number:
    """
    Returns the median (for grouped data) from the discrete frequency distribution $data,
    an already sorted sequence or (value, freq) pairs.
    >>> data = [(0, 1), (1, 6), (2, 7), (3, 2), (4, 3), (5, 1)]
    >>> fmedian(data)
    2.0
    >>> freqs_get_class(dm, fmedian(dm))
    (2, 7)
    """
    counter = Counter()
    for c, f in data:
        counter[c] = f
    return median(list(counter.elements()))

def gmedian (data: IntervalDict) -> Number:
    """
    Returns the median (for grouped data) from the Intervaldict $data.
    >>> i = make_data_intervals_from_freq([(0,3),(5,2),(10,5)], 5, overlap=True)
    >>> i
    {(0, 5): [2.5, 2.5, 2.5], (5, 10): [7.5, 7.5], (10, 15): [12.5, 12.5, 12.5, 12.5, 12.5]}
    >>> gmedian(i)
    10.0
    >>> i = IntervalDict(((0,10),(11,20)), range(0, 20, 3))
    >>> i
    {(0, 10): [0, 3, 6, 9], (11, 20): [12, 15, 18]}
    >>> gmedian(i)
    8.75
    >>> d = Counter({ # adapted from the python doc
    ...         20: 172,   # 20 to 30 years old
    ...         30: 484,   # 30 to 40 years old
    ...         40: 387,   # 40 to 50 years old
    ...         50:  22,   # 50 to 60 years old
    ...         60:   6,   # 60 to 70 years old
    ...     })
    >>> i = make_data_intervals_from_freq(d.items(), 10, overlap=True)
    >>> m = gmedian(i)
    >>> m
    37.510330578512395
    >>> i[m]
    ((30, 40), (35.0, 35.0, 35.0, 35.0, ...))
    >>> cls, freqs = ([4, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0], [10, 18, 22, 25, 40, 15, 10, 8, 7])
    >>> d = {a:b for a,b in zip(cls,freqs)}
    >>> d
    {4: 10, 4.5: 18, 5.0: 22, 5.5: 25, 6.0: 40, 6.5: 15, 7.0: 10, 7.5: 8, 8.0: 7}
    >>> gmedian(make_data_intervals_from_freq(d.items(), 0.5, overlap=True))
    6.03125
    """
    # begin
    observations = data.length
    median_class, cum_freq = median_class_and_cumfreq(data.items(), vfunc=lambda x:len(x))
    class_freq = len(data.get_values(median_class))
    # get the cumulative freq just before the median class:
    cum_freq = cumulative_freq(data.items(), median_class, cmpfunc=lambda x,y:(x[0]<y,len(x[1])))
    lower_class_limit = median_class[0]
    class_size = median_class[1] - median_class[0]
    return lower_class_limit + ( ((observations / 2) - cum_freq) / class_freq ) * class_size


# variance:

def _gvariance (mean: Number,
                data: Sequence[Sequence[Number,Number], ...],
                fromsample: bool) -> Number:
    """
    Returns the variance of $data.
    $mean: mean of $data
    $data: sequence of (value, frequencey) pairs
    $fromsample: if data is from a sample
    """
    tot_score = 0
    tot_freq = 0
    for score, f in data:
        tot_score += f * (deviation(mean, score)**2)
        tot_freq += f
    return tot_score / (tot_freq - (1 if fromsample else 0))

def gvariance (data,
               dtype: GDTYPE,
               fromsample: bool = False) -> Number:
    """
    Returns the variance of $data for grouped data (using the Actual Mean Method).
    $dtype can be one of:
        "basic" | GDTYPE.basic ----------> if $data contains individual measurament values (default)
        "freq" | GDTYPE.freq ------------> if $data is a sequence of (value, frequency) pairs
        "intervals" | GDTYPE.intervals --> if $data is an IntervalDict
    Raise TypeError for unknown values.
    Set $fromsample to True if $data is a sample.
    """
    dt = GDTYPE[dtype]
    data = dt.data2freq(data)
    mean = fgmean(data)
    return _gvariance(mean, data, fromsample)

def gstandard_dev (data,
                   dtype: GDTYPE,
                   fromsample: bool = False) -> Number:
    """
    Returns the standard deviation (using the Actual Mean Method) for grouped data.
    $dtype can be one of:
        "basic" | GDTYPE.basic ----------> if $data contains individual measurament values (default)
        "freq" | GDTYPE.freq ------------> if $data is a sequence of (value, frequency) pairs
        "intervals" | GDTYPE.intervals --> if $data is an IntervalDict
    Raise ValueError for unknown values.
    Set $fromsample to True if $data is a sample.
    """
    return math.sqrt(gvariance(data, dtype, fromsample))

def gstandard_error(data,
                    dtype: GDTYPE,
                    fromsample: bool = False) -> Number:
    """
    Standard error (using the Actual Mean Method) for grouped data.
    $dtype can be one of:
        "basic" | GDTYPE.basic ----------> if $data contains individual measurament values (default)
        "freq" | GDTYPE.freq ------------> if $data is a sequence of (value, frequency) pairs
        "intervals" | GDTYPE.intervals --> if $data is an IntervalDict
    Raise TypeError for unknown values.
    Set $fromsample to True if $data is a sample.
    """
    dt = GDTYPE[dtype]
    data = dt.data2freq(data)
    fsum = sum(f for v, f in data)
    return gstandard_dev(data, 'freq', fromsample) / math.sqrt(fsum)


################
# other stuffs #
################

def print_info (data, issample=False, prefix=None, justify_by=None):
    """Prints info for ungrouped data."""
    if prefix is None:
        prefix = 'sample' if issample else 'population'
    jb = len(prefix) if justify_by is None else justify_by
    print('{:{}} mean: {:.2f} | variance: {:.2f} | st_dev: {:.2f} | st_err {:.2f} | min,max='.format(
            prefix, jb, mean(data), variance(data, issample),
        standard_dev(data, issample), standard_error(data, issample)), min(data), max(data))

def gprint_info (data, dtype, issample=False, prefix=None, justify_by=None):
    """Prints info for grouped data."""
    if prefix is None:
        prefix = 'sample' if issample else 'population'
    jb = len(prefix) if justify_by is None else justify_by
    print('{:{}} mean: {:.2f} | variance: {:.2f} | st_dev: {:.2f} | st_err {:.2f} ({})'.format(
            prefix, jb, gmean(data), gvariance(data, dtype, issample),
        gstandard_dev(data, dtype, issample), gstandard_error(data, dtype, issample), dtype))


def mode (data):
    ...
    c = Counter
    for v in data:
        c[v] += 1
    return max(c.items(), key=lambda x:x[1])

def _test():
    # XXX+TODO: mode
    #mode()


    
    exit()
    
if __name__ == '__main__':
    dx=[(1,10),(2,21),(3,20),(4,26),(5,20),(6,4)]
    d = Counter({ 20: 172, 30: 484,40: 387,50:  22,60:   6, })
    dm = [(0, 1), (1, 6), (2, 7), (3, 2), (4, 3), (5, 1)]
    if 1:_test()
    """
    At the moment, this module can be used as a script for examples purpose only...
    possibly TODO add code to read and manipulate data.
    """
    TOTAL_POPULATION = 10000
    SAMPLE_SIZE = 300
    MINVAL = 1
    MAXVAL = 100
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-N', '--population-size',
                        dest='population_size', type=int, default=TOTAL_POPULATION, metavar='POSITIVE_NUMBER',
                        help='the population size (default: %(default)s)')
    parser.add_argument('-s', '--sample-size',
                        dest='sample_size', type=int, nargs='+', default=[SAMPLE_SIZE], metavar='POSITIVE_NUMBER',
                        help='samples size (default: %(default)s)')
    parser.add_argument('-S', '--sample-repeat',
                        dest='repeat_sample', type=int, default=1, metavar='POSITIVE_NUMBER',
                        help='for each sample size given, repeats a random sample pickup (default: %(default)s)')
    parser.add_argument('-m', '--min',
                        dest='minval', type=int, default=MINVAL, metavar='POSITIVE_NUMBER',
                        help='min value for the population values (default: %(default)s)')
    parser.add_argument('-M', '--max',
                        dest='maxval', type=int, default=MAXVAL, metavar='POSITIVE_NUMBER',
                        help='max value for the population values (default: %(default)s)')
    parser.add_argument('-g', '--grouped',
                        dest='grouped', action='store_true',
                        help='calculate grouped means')
    parsed = parser.parse_args()
    if parsed.population_size < 1: parser.error('wrong population size (must be >= 1)')
    for size in parsed.sample_size:
        if size < 1: parser.error(f'wrong sample size (must be >= 1, got: {size})')
        if parsed.population_size < size: parser.error(f'wrong sample size (bigger then the population: {size})')
    if parsed.repeat_sample < 1: parser.error('wrong sample repetition (must be >= 1)')
    if parsed.minval > parsed.maxval: parser.error('min value > max value')

    # formatting
    pop_prefix = f'[U] population ({parsed.population_size})'
    jb = max(len(pop_prefix), max(len(f'sample ({size})') for size in parsed.sample_size))

    # doit:
    population = get_pop(parsed.population_size, parsed.minval, parsed.maxval)
    print(f"*** population size = {parsed.population_size} | samples size = {parsed.sample_size} | min,max = {parsed.minval},{parsed.maxval}")
    print_info(population, False, pop_prefix, jb)

    # grouped
    if parsed.grouped:
        pop_prefix = f'[G] population ({parsed.population_size})'
        gprint_info(population, GDTYPE.basic, False, pop_prefix, jb)
        data = make_data_freq(population)
        gprint_info(data, GDTYPE.freq, False, pop_prefix, jb)
        pop_classes = autogroup(5, parsed.minval, parsed.maxval)
        data = IntervalDict(pop_classes, population)
        gprint_info(data, GDTYPE.interval, False, pop_prefix, jb)
        
    samples = list(get_sample(population, size) for size in parsed.sample_size for _ in range(parsed.repeat_sample))
    for sample in samples:
        sl = len(sample)
        sample_prefix = f'[U] sample ({sl})'
        print_info(sample, True, sample_prefix, jb)
        # grouped
        if parsed.grouped:
            sample_prefix = f'[G] sample ({sl})'
            gprint_info(sample, GDTYPE.basic, True, sample_prefix, jb)
            data = make_data_freq(sample)
            gprint_info(data, GDTYPE.freq, False, sample_prefix, jb)
            pop_classes = autogroup(5, parsed.minval, parsed.maxval)
            data = IntervalDict(pop_classes, sample)
            gprint_info(data, GDTYPE.interval, False, sample_prefix, jb)


    ############
    # EXAMPLES #
    ############

    EXAMPLES = ''' EXAMPLES:
    
    >>> variance([5,5,5,5])
    0.0
    >>> variance([2,3,4,5])
    1.25
    >>> variance([115,5,5,5])
    2268.75
    >>> variance([115,5,5,-115])
    6618.75

    
    >>> mean([1,2,3,4,5,6])
    3.5
    >> gmean([1,2,3,4,5,6])
    3.5
    >>> mean([1,1,1,1,1,2,3,4,5,6,99])
    11.272727272727273
    >>> gmean([1,1,1,1,1,2,3,4,5,6,99])
    11.272727272727273
    >>> igmean(make_data_intervals([1,1,1,1,1,2,3,4,5,6,99], [(0,10),(10,20),(20,100)]))
    10.0

    
    >>> make_data_intervals(chain(*[range(10),range(1,10,3), range(1,10,2),(0,1,2)]), [(0,5),(6,10)])
    {(0, 5): [0, 1, 2, 3, 4, 5, 1, 4, 1, 3, 5, 0, 1, 2], (6, 10): [6, 7, 8, 9, 7, 7, 9]}
    >>> freqs_from_intervals(make_data_intervals(chain(*[range(10),range(1,10,3), range(1,10,2),(0,1,2)]), [(0,5),(6,10)]))
    [(2.5, 14), (8.0, 7)]
    >>> fgmean(freqs_from_intervals(make_data_intervals(chain(*[range(10),range(1,10,3), range(1,10,2),(0,1,2)]), [(0,5),(6,10)])))
    4.333333333333333
    >>> igmean(make_data_intervals(chain(*[range(10),range(1,10,3), range(1,10,2),(0,1,2)]), [(0,5),(6,10)]))
    4.333333333333333

    
    >>> data = list(sorted(chain(*[range(20),range(1,20,3), range(1,20,2),[0,1,2]*5])))
    >>> data
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 6,
     7, 7, 7, 8, 9, 9, 10, 10, 11, 11, 12, 13, 13, 13, 14, 15, 15, 16, 16, 17, 17, 18, 19, 19, 19]
    >>> i = make_data_intervals(data, [(0,5),(6,10),(10,15),(16,20)])
    >>> i
    {(0, 5): [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5],
     (6, 10): [6, 7, 7, 7, 8, 9, 9, 10, 10],
     (10, 15): [11, 11, 12, 13, 13, 13, 14, 15, 15],
     (16, 20): [16, 16, 17, 17, 18, 19, 19, 19]}
    >>> type(i)
    <class '__main__.IntervalDict'>
    >>> gvariance(i, GDTYPE.interval)
    33.82720044378698
    >>> gstandard_dev(i, GDTYPE.interval)
    5.8161155803325455
    >>> gstandard_error(i, GDTYPE.interval)
    0.8065501134197689
    >>> gstandard_error(i, GDTYPE.interval, fromsample=True)
    0.8144190813544674


    >>> # https://flexbooks.ck12.org/cbook/ck-12-cbse-math-class-10/section/14.3/primary/lesson/median-of-grouped-data/
    >>> list(range(0,50,10))
    [0, 10, 20, 30, 40]
    >>> d={a:b for a,b in zip(x,[2,4,5,4,2])}
    >>> gmedian(make_data_intervals_from_freq(d.items(),10,overlap=True))
    25.0
    >>> from mismatched_socks import frange
    >>> cls = list(frange(4,8.5,0.5))
    >>> freqs = [10,18,22,25,40,15,10,8,7]
    >>> cls, freqs
    ([4, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0], [10, 18, 22, 25, 40, 15, 10, 8, 7])
    >>> d = {a:b for a,b in zip(cls,freqs)}
    >>> d
    {4: 10, 4.5: 18, 5.0: 22, 5.5: 25, 6.0: 40, 6.5: 15, 7.0: 10, 7.5: 8, 8.0: 7}
    >>> gmedian(make_data_intervals_from_freq(d.items(),0.5,overlap=True))
    6.03125
    >>> # also:
    >>> i = make_data_intervals_from_freq(d.items(),0.5,overlap=True)
    >>> m = gmedian(i)
    >>> m
    6.03125
    >>> i[m]
    ((6.0, 6.5), (6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25, 6.25))

    
    # unequal classes:
    >>> i = IntervalDict(overlap=True)
    >>> i.add_interval((0,10),[5]*8)
    >>> i.add_interval((10,30),[20]*20)
    >>> i.add_interval((30,60),[46]*36)
    >>> i.add_interval((60,80),[70]*24)
    >>> i.add_interval((80,90),[85]*12)
    >>> i
    {(0, 10): [5, 5, 5, 5, 5, 5, 5, 5], (10, 30): [20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20], (30, 60): [46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46], (60, 80): [70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70, 70], (80, 90): [85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85, 85]}
    >>> gmedian(i)
    48.333333333333336
    >>> i[gmedian(i)]
    ((30, 60), (46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46, 46))

    '''
