#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Some statistical utilities

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

"""
Statistical stuff.

HAS_SYSTEMRANDOM is True when your system supports random.SystemRandom

"""
import math
import random
try:
    from random import SystemRandom
    HAS_SYSTEMRANDOM_EXC = ''
    HAS_SYSTEMRANDOM = True
    class RandomRange(random.SystemRandom):
        def __init__ (self, min_value, max_value):
            if min_value >= max_value:
                raise ValueError(f'min_value ({min_value}) >= max_value ({max_value})')
            self.min_value = min_value
            self.max_value = max_value
            super().__init__(self)
        def randrange (self):
            return int(self.random()
                       * (10 ** (math.floor(math.log10(self.max_value)) + 1))
                       % (self.max_value - self.min_value)) + self.min_value
except ImportError as e:
    HAS_SYSTEMRANDOM_EXC = e
    HAS_SYSTEMRANDOM = False
    # No random.SystemRandom, backup to random.randrange
    class RandomRange:
        def __init__ (self, min_value, max_value):
            if min_value >= max_value:
                raise ValueError(f'min_value ({min_value}) >= max_value ({max_value})')
            self.min_value = min_value
            self.max_value = max_value
        def randrange (self):
            return random.randrange(self.min_value, self.max_value)

def getrand (minval, maxval, repeat):
    """Yields random values between $minval and $maxval (inclusive) $repeat times.
    Uses random.SystemRandom when available, otherwise random.randrange."""
    rr = RandomRange(minval, maxval)
    for _ in range(repeat):
        yield rr.randrange()

#########
# TESTS #
#########

def _test (minval, maxval, repeat, verbose=False, end=' '):
    #if verbose: print(f'min: {minval} | max: {maxval} | repeat: {repeat} | verbose: {verbose}')
    result = True
    tail_print = True
    for rv in getrand(minval, maxval, repeat):
        try:
            assert rv >= minval and rv <= maxval
            if verbose:
                print(rv, end=end)
        except AssertionError as e:
            print(f"ERROR: min: {minval} | max: {maxval} | rv: {rv}")
            result = False
            tail_print = False
    if verbose:
        print() if tail_print else None
    return result
    
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('min', type=int, help='range min value')
    p.add_argument('max', type=int, help='range max value')
    p.add_argument('-r', '--repeat', type=int, default=1000, metavar='N', help='repeat random extraction %(metavar)s times (default: %(default)s)')
    p.add_argument('-s', '--separator', default='\n', metavar='STR', help='use %(metavar)s as separator (default: "\\n")' )
    p.add_argument('-t', '--test', action='store_true', help='perform some test and exit.')
    p.add_argument('-v', '--verbose', action='store_true', default=False, help='show random values (during tests)')
    p.add_argument('-w', '--warn', action='store_true', default=False, help='show warnings (check random.SystemRandom)')
    parsed = p.parse_args()
    if not HAS_SYSTEMRANDOM and parsed.warn:
        # for no particular reason ^L^
        (lambda m = __import__('warnings'): setattr(
            m, 'showwarning', lambda w, *t, **f: print(w, file=__import__('sys').stderr))
         or  m.warn)()(f'WARNING: {HAS_SYSTEMRANDOM_EXC}: Backup to random.randrange')
        #import sys, warnings
        #warnings.showwarning = lambda msg, *a, **k: print(msg, file=sys.stderr)
        #warnings.warn(f'WARNING: {HAS_SYSTEMRANDOM_EXC}: Backup to random.randrange')
    if parsed.min >= parsed.max:
        p.error(f'min value too big: min: {parsed.min} | max: {parsed.max}')
    if parsed.repeat < 1:
        p.error('repeat must be >= 1')
    if parsed.test:
        _test(parsed.min, parsed.max, parsed.repeat, parsed.verbose)
    else:
        for value in getrand(parsed.min, parsed.max, parsed.repeat):
            print(value, end=parsed.separator)
        if parsed.separator != '\n':
            print()
