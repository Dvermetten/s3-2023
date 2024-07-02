# Copyright (C) 2023 Alexandre Jesus <https://adbjesus.com>, Carlos M. Fonseca <cmfonsec@dei.uc.pt>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

from __future__ import annotations

from typing import Protocol, TypeVar, Any, Optional, Tuple
from collections.abc import Iterable, Callable, Sequence

from operator import itemgetter
from itertools import tee
from math import ceil, log2
import math
import random

T = TypeVar('T')

def argmax(v: Iterable[T]) -> int:
    return max(enumerate(v), key=itemgetter(1))[0]

def argmin(v: Iterable[T]) -> int:
    return min(enumerate(v), key=itemgetter(1))[0]

def or_default(v: Optional[T], default: Callable[[], T]) -> T:
    return v if v is not None else default()

def non_repeating_lcg(n: int, seed: Optional[int] = None) -> Iterable[int]:
    if seed is not None:
        random.seed(seed)
    "Pseudorandom sampling without replacement in O(1) space"
    if n > 0:
        a = 5 # always 5
        m = 1 << ceil(log2(n))
        if m > 1:
            c = random.randrange(1, m, 2)
            x = random.randrange(m)
            for _ in range(m):
                if x < n: yield x
                x = (a * x + c) % m
        else:
            yield 0

def sample(n: int, seed: Optional[int] = None) -> Iterable[int]:
    for v in non_repeating_lcg(n, seed):
        yield v

def sample2(n: int, m: int, seed: Optional[int] = None) -> Iterable[Tuple[int, int]]:
    for v in non_repeating_lcg(n*m, seed):
        i = v // m
        j = v % m
        yield (i, j)

def isclose(a, b, rel_tol = 1e-6, abs_tol = 1e-9):
    return math.isclose(a, b, rel_tol = rel_tol, abs_tol = abs_tol)

# FIXME(3.10): This can be replaced by itertools.pairwise if we use
# python 3.10
def pairwise(iterable: Iterable[T]) -> Iterable[Tuple[T, T]]:
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def parse_parameters(configuration_parameters: Sequence[str]) -> dict:
    """
    Function to parse algorithm parameters (following irace specifications) into dictionaries to be used in the target-runner

    Parameters
    ----------
    configuration_parameters:
        The command line arguments, without the executables name
    Notes
    -----
    Returns Dictionary with parsed parameters
    """
    import argparse 

    parser = argparse.ArgumentParser(
        description="Run basic tuning of specified algorithm",
        argument_default=argparse.SUPPRESS,
    )

    # Positional parameters (defined by irace, need to be present)
    parser.add_argument("configuration_id", type=str)
    parser.add_argument("instance_name", type=str)
    parser.add_argument("seed", type=int)
    parser.add_argument("instance", type=argparse.FileType('r'))
    # parser.add_argument("bound", type=float) #Disabled, for now we use cbudget and lbudget as fixed parameters instead

    # algorithm parameters
    parser.add_argument("--csearch", dest="csearch", type=str, required=True)
    parser.add_argument("--lsearch", dest="lsearch", type=str, required=True)
    parser.add_argument("--alpha", dest="alpha", type=float, default=0.01)
    parser.add_argument("--beta", dest="beta", type=float, default=5.0)
    parser.add_argument("--rho", dest="rho", type=float, default=0.5)
    parser.add_argument("--tau0", dest="tau0", type=float, default=1/3000.0)
    parser.add_argument("--taumax", dest="taumax", type=float, default=1/3000.0)
    parser.add_argument("--globalratio", dest="globalratio", type=float, default=0.5)
    parser.add_argument("--init_temp", dest="init_temp", type=float, default=30.0)
    parser.add_argument("--cbudget", dest="cbudget", type=float, default=1)
    parser.add_argument("--lbudget", dest="lbudget", type=float, default=1)

    # Process into dicts
    return parser.parse_args(configuration_parameters).__dict__
