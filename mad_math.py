#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Some math utilities

# Copyright (C) 2025  Marco Chieppa | crap0101

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

from collections import defaultdict
from functools import reduce
from math import gcd, sqrt
from operator import mul


def avg (seq):
    """Get the average of $seq"""
    tot = 0
    for n, v in enumerate(seq, 1):
        tot += v
    try:
        return tot / n
    except UnboundLocalError:
        return 0


def dec2bin(n):
    """return a bit-string representation of *n*."""
    b = [str(n & 1)]
    while (n := n >> 1):
        b.append(str(n & 1))
    return ''.join(map(str, reversed(b)))


def decimal_threshold (n, precision=4):
    """To consider equal a number $n and its
    integral value under a certain $precision."""
    exp = 10 ** precision
    n = abs(n)
    r =  (n - int(n)) * exp
    return True if (n - int(n)) * exp < 1 else False


def eqd (a, b, delta, precision=4):
    """
    Return True if number $a and $b are equals considered
    the given $delta and $precision.
    Raise ValueError for $precision values < 0 (default is 4).
    $precision is truncated to the nearest integral toward 0.
    """
    if precision < 0:
        raise 
    a = abs(a)
    b = abs(b)
    exp = 10 ** int(precision)
    return (abs((a - b) * exp) <= abs(delta * exp)) 


def is_prime (x):
    """Return True if *x* is a prime number, False otherwise."""
    if x < 2 or (x != 2 and not x % 2):
        return False
    elif x in (2, 3, 5, 7):
        return True
    a = 3
    limit = sqrt(x)
    while a <= limit:
        if not (x % a):
            return False
        a += 2
    return True


def primes_from (n):
    """Yield primes biggers than *n*."""
    n += 1
    if not n % 2:
        n += 1
    while True:
        if is_prime(n):
            yield n
        n += 2


def next_prime (n):
    """Return the first prime number bigger than *n*."""
    return next(primes_from(n))


def perc (value, perc, fun=lambda n:n):
    """
    value => a number
    perc  => the requested percentage of $value to get
    fun   => a callable to be applied to the result
             (like int(), math.floor(), ...).
             Default to the identity function.
    """
    x = perc * value / 100
    return fun(x)


def in_perc_range (num, value, perc_value, fun=lambda n:n):
    """
    Returns True if $num is in the ±$perc_value range of $value.
    >>> in_perc_range(110, 100, 10)
    True
    >>> in_perc_range(111, 100, 10)
    False
    >>> in_perc_range(90, 100, 10)
    True
    >>> in_perc_range(89, 100, 10)
    False
    """
    x = perc(value, perc_value, fun)
    return num >= (value - x) and num <= (value + x)


def prime_factors (n):
    """Return a list of prime factors of *n*."""
    lst = []
    actual = 2
    limit = int(sqrt(n))
    while actual <= limit:
        if not n % actual:
            n //= actual
            limit = sqrt(n)
            lst.append(actual)
        else:
            actual += 1 + actual % 2
    if n > 1:
        lst.append(n)
    return lst


def prime_factors_dict (n):
    """
    Return the prime factors of *n* as a dict of base:exp items.
    """
    factors = defaultdict(int)
    actual = 2
    limit = int(sqrt(n))
    while actual <= limit:
        if not n % actual:
            n //= actual
            limit = sqrt(n)
            factors[actual] += 1
        else:
            actual += 1 + actual % 2
    if n > 1:
        factors[n] += 1
    return factors


def prime_factors_i (num):
    """ Yields the prime factors of *num*."""
    num = num
    actual = 2
    limit = int(sqrt(num))
    while actual <= limit:
        if not num % actual:
            num //= actual
            limit = sqrt(num)
            yield actual
        else:
            actual += 1 + (actual % 2)
    if num > 1:
        yield num


def totient (n):
    """Return the Euler's totient of *n*."""
    return sum(1 for x in range(1, n+1) if gcd(x, n) == 1)


def totient_pairs (n):
    """Return pairs of (x, n) in range (x=1, x=n+1) for which gcd(x, n) == 1."""
    return list((x, n) for x in range(1, n+1) if gcd(x, n) == 1)


###################### TODO: tests

def _test_avg():
    assert avg([]) == 0, f'FAIL: avg([])'
    assert avg([0]) == 0, f'FAIL: avg([0])'
    assert avg([1]) == 1, f'FAIL: avg([1])'
    assert avg([0, 10]) == 5, f'FAIL: avg([0, 10])'
    return True

def _test_primes():
    for i in range(2, 100000):
        factors = prime_factors(i)
        assert (reduce(mul, factors)
                == reduce(mul, (k**v for k, v in prime_factors_dict(i).items()))
                == reduce(mul, prime_factors(i))), f'FAIL [factors]: {i}'
        for f in factors:
            assert is_prime(f) == True, f'FAIL: is_prime({f})'
            assert is_prime(f + 7) == False, f'FAIL: is_prime({f+7})'
    return True

def _test_bin():
    for i in range(10000):
        assert bin(i)[2:] == dec2bin(i)
    return True

def _test_perc():
    for i in range(11):
        assert True == in_perc_range(100+i, 100, 10), f'FAIL: in_per_range: {(100+i,100,10)}'
        assert False == in_perc_range(111+i, 100, 10), f'FAIL: in_per_range: {(111+i,100,10)}'
        assert True == in_perc_range(90+i, 100, 10), f'FAIL: in_per_range: {(90+i,100,10)}'
        assert False == in_perc_range(89-i, 100, 10), f'FAIL: in_per_range: {(89-i,100,10)}'
    assert True == in_perc_range(80, 100, 20), f'FAIL: in_per_range: {(80,100,20)}'
    assert True == in_perc_range(40, 50, 20), f'FAIL: in_per_range: {(40,50,20)}'
    return True

def _test_threshold():
    n = 1.00005
    for i in range(5):
        assert True == decimal_threshold(n, i), f'FAIL: decimal_threshold({n}{i})'
    for i in range(5, 8):
        assert False == decimal_threshold(n, i), f'FAIL: decimal_threshold({n}{i})'
    return True

def _test_eqd():
    from random import randint
    tuples = [
        (False, 22, 24, 1),
        (True, 22, 23, 1),
        (True, 22, 23, 3),
        (False, 22.009,22.007,0.001),
        (True, 22.009,22.007,0.002),
        (True, 22.009,22.007,0.003),
    ]
    for i in range(10,100, 13):
        a = randint(i, i+100) / randint(2, i-1)
        b = randint(i, i+100) / randint(2, i-1)
        delta = a - b
        assert True == eqd(a, b, delta), f'FAIL: eqd({a}, {b}, {delta})'
    assert True == eqd(1.0002, 1.0010, 0.8), f'FAIL: eqd({a}, {b}, {delta})'
    for r, a, b, d in tuples:
        assert r == eqd(a, b, d), f'FAIL: eqd({a}, {b}, {d})'
    return True

def _run_tests():
    _test_avg() and print('Test avg: OK')
    _test_primes() and print('Test is_prime|prime_factors|prime_factors_dict|prime_factors_i: OK')
    _test_bin() and print('Test dec2bin: OK')
    _test_perc() and print('Test in_perc_range: OK')
    _test_threshold() and print('Test decimal_threshold: OK')
    _test_eqd() and print('Test eqd: OK')

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('-t', '--test', dest='test', action='store_true',
                 help='Run tests.')
    args = p.parse_args()
    if args.test:
        _run_tests()
