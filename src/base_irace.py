#!/usr/bin/env python3
#
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

from typing import TextIO, Optional, Any
from collections.abc import Iterable, Hashable

import logging

Objective = Any

class Component:
    @property
    def cid(self) -> Hashable:
        raise NotImplementedError

class LocalMove:
    ...

class Solution:
    def output(self) -> str:
        """
        Generate the output string for this solution
        """
        raise NotImplementedError

    def copy(self) -> Solution:
        """
        Return a copy of this solution.

        Note: changes to the copy must not affect the original
        solution. However, this does not need to be a deepcopy.
        """
        raise NotImplementedError

    def is_feasible(self) -> bool:
        """
        Return whether the solution is feasible or not
        """
        raise NotImplementedError

    def objective(self) -> Optional[Objective]:
        """
        Return the objective value for this solution if defined, otherwise
        should return None
        """
        raise NotImplementedError

    def lower_bound(self) -> Optional[Objective]:
        """
        Return the lower bound value for this solution if defined,
        otherwise return None
        """
        raise NotImplementedError

    def add_moves(self) -> Iterable[Component]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all components that can be added to the solution
        """
        raise NotImplementedError

    def local_moves(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves that can be applied to the solution
        """
        raise NotImplementedError

    def random_local_move(self) -> Optional[LocalMove]:
        """
        Return a random local move that can be applied to the solution.

        Note: repeated calls to this method may return the same
        local move.
        """
        raise NotImplementedError

    def random_local_moves_wor(self) -> Iterable[LocalMove]:
        """
        Return an iterable (generator, iterator, or iterable object)
        over all local moves (in random order) that can be applied to
        the solution.
        """
        raise NotImplementedError
            
    def heuristic_add_move(self) -> Optional[Component]:
        """
        Return the next component to be added based on some heuristic
        rule.
        """
        raise NotImplementedError

    def add(self, component: Component) -> None:
        """
        Add a component to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        raise NotImplementedError

    def step(self, lmove: LocalMove) -> None:
        """
        Apply a local move to the solution.

        Note: this invalidates any previously generated components and
        local moves.
        """
        raise NotImplementedError

    def objective_incr_local(self, lmove: LocalMove) -> Optional[Objective]:
        """
        Return the objective value increment resulting from applying a
        local move. If the objective value is not defined after
        applying the local move return None.
        """
        raise NotImplementedError

    def lower_bound_incr_add(self, component: Component) -> Optional[Objective]:
        """
        Return the lower bound increment resulting from adding a
        component. If the lower bound is not defined after adding the
        component return None.
        """
        raise NotImplementedError

    def perturb(self, ks: int) -> None:
        """
        Perturb the solution in place. The amount of perturbation is
        controlled by the parameter ks (kick strength)
        """
        raise NotImplementedError

    def components(self) -> Iterable[Component]:
        """
        Returns an iterable to the components of a solution
        """
        raise NotImplementedError

class Problem:
    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        """
        Create a problem from a text I/O source `f`
        """
        raise NotImplementedError

    def empty_solution(self) -> Solution:
        """
        Create an empty solution (i.e. with no components).
        """
        raise NotImplementedError


if __name__ == "__main__":
    from api.solvers import *
    from time import perf_counter
    import sys
    from api.utils import parse_parameters

    args_dict = parse_parameters(sys.argv[1:])

    p = Problem.from_textio(args_dict['instance'])
    s: Optional[Solution] = p.empty_solution()

    start = perf_counter()

    if s is not None:
        if args_dict['csearch'] == 'beam':
            s = beam_search(s, 10)
        elif args_dict['csearch'] == 'grasp':
            s = grasp(s, args_dict['cbudget'], alpha = args_dict['alpha'])
        elif args_dict['csearch'] == 'greedy':
            s = greedy_construction(s)
        elif args_dict['csearch'] == 'heuristic':
            s = heuristic_construction(s)
        elif args_dict['csearch'] == 'as':
            ants = [p.empty_solution_with_start(i) for i in range(p.nnodes)]
            s = ant_system(ants, args_dict['cbudget'], beta = args_dict['beta'], rho = args_dict['rho'], tau0 = args_dict['tau0'])
        elif args_dict['csearch'] == 'mmas':
            ants = [p.empty_solution_with_start(i) for i in range(p.nnodes)]
            s = mmas(ants, args_dict['cbudget'], beta = args_dict['beta'], rho = args_dict['rho'], taumax = args_dict['taumax'], globalratio = args_dict['globalratio'])

    if s is not None:
        if args_dict['lsearch'] == 'bi':
            s = best_improvement(s, args_dict['lbudget'])
        elif args_dict['lsearch'] == 'fi':
            s = first_improvement(s, args_dict['lbudget'])
        elif args_dict['lsearch'] == 'ils':
            s = ils(s, args_dict['lbudget'])
        elif args_dict['lsearch'] == 'rls':
            s = rls(s, args_dict['lbudget'])
        elif args_dict['lsearch'] == 'sa':
            s = sa(s, args_dict['lbudget'], args_dict['init_temp'])

    end = perf_counter()

    if s is not None:
        print(s.objective())
    else:
        print("99999999") #placeholder
    sys.exit(0)