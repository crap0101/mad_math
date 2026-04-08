
from collections import defaultdict
from collections.abc import Sequence, ItemsView
import math
import random
from typing import Generic
#
import inspect
# local imports
from mad_math import rand

#############
# some defs #
#############

Number = float | int

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
####################

def autogroup (data: Sequence[Number, ...]|None,
               chunks: None|Number = None,
               minvalue: None|Number = None,
               maxvalue: None|Number = None) -> Sequence[Sequence[Number,Number], ...]:
    """
    Returns class intervals for $data in n $chunks from $minvalue to $maxvalue.
    If the optional arguments are None, some values are extracted from *data* and other guessed.
    $data can also be None, in which case the other arguments must be provided (raises ValueError otherwise).
    """
    if data is None and None in (chunks, minvalue, maxvalue):
        raise ValueError("not enought argument or of the wrong type")
    if chunks is None:
        ld = len(data)
        chunks = int(ld / (ld * .2))
        if chunks < 1:
            raise ValueError("can't make chunks from data")
    if minvalue is None:
        minvalue = min(data)
    if maxvalue is None:
        maxvalue = max(data)
    q = maxvalue - minvalue
    nc = int(q / chunks)
    classes = []
    while minvalue < maxvalue:
        if minvalue + nc >= maxvalue:
            classes.append((minvalue, maxvalue))
            break
        else:
            classes.append((minvalue, minvalue + nc - 1))
            minvalue += nc
    return classes
    #XXX[1]: see othersXXX[1]: use "< imax" for a canonical intervals representation? -- in IntervalDict:
    #     make class range e.g. [(0,10),(10,20),(20,30),...] ? subsequently excluding the upper value?

class IntervalError(Exception):
    """Base error class for IntevalDict objects."""
    def __init__ (self, msg):
        self.msg = msg
        super().__init__(msg)
    def __str__ (self):
        return self.msg

class IntervalDict:
    """
    Object for class intervals.
    """
    def __init__ (self,
                  intervals: Sequence[Sequence[Number,Number], ...] = ((float('-inf'), float('+inf'))),
                  data: None|Sequence[Number, ...] = None,
                  trim : bool = False):
        """
        $intervals is a sequence of (min, max) pairs, filled with the optional values from $data.
        If $trim is True, ignores values from $data which don't fit in the interval, otherwise raises a IntervalError.
        >>> IntervalDict(((0,10),(11,30)), range(5,40,4), True)
        {(0, 10): [5, 9], (11, 30): [13, 17, 21, 25, 29]}
        """
        self.intervals = tuple(intervals)
        self._dict = {tuple(i):[] for i in self.intervals}
        if data:
            self.extend(data, not bool(trim))     

    def __getitem__ (self, value: Number) -> Sequence[Number,Number]:
        """
        Returns the interval's values in which $value *may* belong or raises IntervalError.
        >>> i = IntervalDict(((0,11),(11,30),(31,40)), range(40))
        >>> i[15]
        (12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30)
        {(0, 10): [5, 8], (11, 30): [11, 14, 17, 20, 23]}
        >>> i[7] # 7 is not in the interval's values, but can be. Use get_class to get the interval for existing values only.
        (5, 8)
        """
        for (imin, imax), seq in self._dict.items():
            if value >= imin and value <= imax: #XXX[1]: use "< imax" for a canonical intervals representation?
                return tuple(seq)
        raise IntervalError(f"value '{value}' not in intervals")

    def get_class (self, value: Number) -> Sequence[Number,Number]:
        """
        Returns the class interval in which $value belong or raises IntervalError.
        >>> i = IntervalDict(((0,11),(11,30),(31,40)), range(5,40,4))
        >>> i
        {(0, 11): [5, 9], (11, 30): [13, 17, 21, 25, 29], (31, 40): [33, 37]}
        >>> i.get_class(17)
        (11, 30)
        >>> i.get_class(7)
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
        >>> i.get_class(15)
        Traceback (most recent call last):
        ...
        IntervalError: value '15' not in intervals
        >>> i.get_class(5)
        (0, 10)
        >>> i.get_values(i.get_class(5))
        (5, 9)
        """
        target = tuple(cls)
        for interval, seq in self._dict.items():
            if tuple(interval) == target:
                return tuple(seq)
        raise IntervalError(f"class '{cls}' not found")

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
            if value >= imin and value <= imax: #XXX[1]: use "< imax" for a canonical intervals representation?
                break
        else:
            raise IntervalError(f"value '{value}' not in intervals")
        del self._dict[(imin, imax)]

    def __iter__ (self):
        """NotImplemented"""
        raise NotImplementedError(f"{self.__class__.__name__} {inspect.getframeinfo(inspect.currentframe()).function}")

    def __len__ (self) -> Number:
        """Returns the number of intervals."""
        return len(self._dict)

    def length (self) -> Number:
        """Returns the number of values in all the intervals."""
        return sum(len(v) for v in self._dict.values())

    def __iadd__ (self, value: Number) -> None:
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
            if value >= imin and value <= imax: #XXX[1]: use "< imax" for a canonical intervals representation?
                self._dict[(imin, imax)].append(value)
                return self
        raise IntervalError(f'''value "{value}" doesn't belong to any interval''')

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
        self._dict = dict(sorted(self._dict.items()))

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

    def items (self) -> ItemsView:
        """Returns a new view of the underling dictionary's items."""
        return self._dict.items()

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
        Removes the interval in which $value belong or raises IntervalError.
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


def make_data_intervals (data: Sequence[Number],
                         intervals: Sequence[[Number,Number]]) -> IntervalDict:
    """
    Returns an IntervalDict with the given $intervals fillen with values from $data.
    >>> make_data_intervals([0,1,2], [(0,5),(6,10)])
    {(0, 5): [0, 1, 2], (6, 10): []}
    """
    d = IntervalDict(intervals)
    for value in data:
        d += value
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

def freqs_from_intervals (data: IntervalDict) -> Sequence[[Number,Number]]:
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

def fgmean (data: Sequence[[Number,Number]]) -> Number:
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
        mean = ugmean
        data2freq = make_data_freq

    class freq(metaclass=StrCls):
        name = 'freq'
        mean = fgmean
        data2freq = lambda x: x

    class interval(metaclass=StrCls):
        name = 'interval'
        mean = igmean
        data2freq = freqs_from_intervals

    def __class_getitem__(cls, item):
        if item in ('basic', cls.basic):
            return cls.basic
        elif item in ('freq', cls.freq):
            return cls.freq
        elif item in ('interval', cls.interval):
            return cls.interval
        else:
            raise TypeError(f"No such item '{item}'")

# variance:

def _gvariance (mean: Number,
                data: Sequence[[Number,Number]],
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

def gvariance (data, dtype, fromsample: bool = False) -> Number:
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

def gstandard_dev (data, dtype, fromsample: bool = False) -> Number:
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

def gstandard_error(data, dtype, fromsample: bool = False) -> Number:
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
            prefix, jb, mean(data), variance(data, issample), standard_dev(data, issample), standard_error(data, issample)), min(data), max(data))

def gprint_info (data, dtype, issample=False, prefix=None, justify_by=None):
    """Prints info for grouped data."""
    if prefix is None:
        prefix = 'sample' if issample else 'population'
    jb = len(prefix) if justify_by is None else justify_by
    print('{:{}} mean: {:.2f} | variance: {:.2f} | st_dev: {:.2f} | st_err {:.2f} ({})'.format(
            prefix, jb, gmean(data), gvariance(data, dtype, issample), gstandard_dev(data, dtype, issample), gstandard_error(data, dtype, issample), dtype))

def _test():
    from statistics import median_grouped, median
    from collections import Counter
    demographics = Counter({
    25: 172,   # 20 to 30 years old
    35: 484,   # 30 to 40 years old
    45: 387,   # 40 to 50 years old
    55:  22,   # 50 to 60 years old
    65:   6,   # 60 to 70 years old
    })
    '''
    >>> data = list(demographics.elements())
    >>> median(data)
    35
    >>> round(median_grouped(data, interval=10), 1)
    37.5
    '''
    def gmedian (data, **k):
        tot = sum(data.values()) / 2
        p = 0
        for k, v in data.items():
            p += v
            if p >= tot:
                return k
        raise ValueError("Not found")
    print("gmedian:", gmedian(demographics))
    exit()
    
if __name__ == '__main__':
    """
    At the moment, this module can be used as a script for examples pourpose only...
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

    # doit
    
    population = get_pop(parsed.population_size, parsed.minval, parsed.maxval)
    print(f"*** population size = {parsed.population_size} | samples size = {parsed.sample_size} | min,max = {parsed.minval},{parsed.maxval}")
    print_info(population, False, pop_prefix, jb)

    # grouped
    if parsed.grouped:
        pop_prefix = f'[G] population ({parsed.population_size})'
        gprint_info(population, GDTYPE.basic, False, pop_prefix, jb)
        data = make_data_freq(population)
        gprint_info(data, GDTYPE.freq, False, pop_prefix, jb)
        pop_classes = autogroup(population, 5, parsed.minval, parsed.maxval)
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
            pop_classes = autogroup(sample, 5, parsed.minval, parsed.maxval)
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
    '''
