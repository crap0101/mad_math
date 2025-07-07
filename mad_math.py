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
from random import shuffle


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

def mad_max (iterable, default=[], key=lambda x:x):
    """
    Return the biggest ITEMS from *iterable*.
    *default* specifies an object to return if
    the provided iterable is empty.
    *key* is a callable applied on every item, the results
    of key(item) is used for the comparison (default to
    the identity function).
    """
    iterable = iter(iterable)
    founds = []
    try:
        first = next(iterable)
        max = key(first)
        founds.append(first)
    except StopIteration:
        return default
    for item in iterable:
        this = key(item)
        if this > max:
            max = this
            founds = [item]
        elif max == this:
            founds.append(item)
    return founds


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


###################### TODO: tests
def _test_primes():
    for i in range(2, 100000):
        factors = prime_factors(i)
        assert (reduce(mul, factors)
                == reduce(mul, (k**v for k, v in prime_factors_dict(i).items()))
                == reduce(mul, prime_factors(i))), f'FAIL [factors]: {i}'
        for f in factors:
            assert is_prime(f) == True, f'FAIL: is_prime({f})'
            assert is_prime(f + 7) == False, f'FAIL: is_prime({f+13})'
    return True

def _test_max():
    limit = 10000
    l1 = list(range(1, limit))
    l2 = l1.copy()
    shuffle(l2)
    l1.extend(l2)
    lmax = mad_max(l1)
    assert len(lmax) == 2, f'FAIL: mad_max: wrong len|'
    assert len(set(lmax)) == 1, f'FAIL: mad_max: values differs!'
    assert lmax[0] == limit - 1, f'FAIL: mad_max: not really the max!'
    return True


if __name__ == '__main__':
    _test_primes() and print('Test is_prime|prime_factors|prime_factors_dict|prime_factors_i: OK')
    _test_max() and print('Test mad_max: OK')
