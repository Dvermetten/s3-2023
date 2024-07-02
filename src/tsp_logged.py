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

from typing import TextIO, Optional, cast, NewType
from collections.abc import Sequence, Iterable, Hashable

from dataclasses import dataclass
from math import sqrt, isclose
from copy import copy
from itertools import chain
import random
import logging
import sys

from api.utils import or_default, pairwise, sample2

import pandas as pd


class BasicLogger:
    def __init__(self, algname: Optional[str] = None, fid: Optional[int] = None):
        self.records = []
        self.fid = fid
        self.algname = algname
        self.start_time = perf_counter()

    def __call__(self, y):
        self.records.append([perf_counter() - self.start_time, y])

    def write(self, fname: str):
        dt = pd.DataFrame.from_records(
            self.records, columns=["runtime", "fval"])
        if self.algname:
            dt["algname"] = self.algname
        if self.fid:
            dt["fid"] = self.fid
        dt["nanoseconds"] = dt["runtime"] * 1_000_000
        dt["nanoseconds"] = dt["nanoseconds"].astype(int)
        dt.to_csv(fname, index=False)


@dataclass
class Component:
    u: int
    v: int

    @property
    def cid(self) -> Hashable:
        return self.u, self.v


@dataclass
class LocalMove:
    i: int
    j: int


if sys.version_info < (3, 9):
    Path = NewType("Path", list)
    Used = NewType("Used", set)
    Unused = NewType("Unused", set)
else:
    Path = NewType("Path", list[int])
    Used = NewType("Used", set[int])
    Unused = NewType("Unused", set[int])


class Solution:
    def __init__(
        self,
        problem: Problem,
        start: int,
        path: Path,
        used: Used,
        unused: Unused,
        dist: float,
        logger: Optional[BasicLogger] = None,
    ) -> None:
        self.problem = problem
        self.start = start
        self.path = path
        self.used = used
        self.unused = unused
        self.dist = dist
        self.logger = logger

    def output(self) -> str:
        return "\n".join(map(str, self.path))

    def copy(self):
        return self.__class__(
            self.problem,
            self.start,
            copy(self.path),
            copy(self.used),
            copy(self.unused),
            self.dist,
            self.logger,
        )

    def is_feasible(self) -> bool:
        return len(self.path) == self.problem.nnodes + 1

    def objective(self) -> Optional[float]:
        if len(self.path) == self.problem.nnodes + 1:
            if self.logger:
                self.logger(self.dist)
            return self.dist
        else:
            return None

    def lower_bound(self) -> Optional[float]:
        return self.dist

    def add_moves(self) -> Iterable[Component]:
        if len(self.path) < self.problem.nnodes:
            u = self.path[-1]
            for v in self.unused:
                yield Component(u, v)
        elif len(self.path) == self.problem.nnodes:
            u = self.path[-1]
            yield Component(u, self.start)

    def local_moves(self) -> Iterable[LocalMove]:
        for i in range(1, len(self.path)):
            for j in range(i + 2, len(self.path)):
                yield LocalMove(i, j)

    def random_local_move(self) -> Optional[LocalMove]:
        if len(self.path) >= 4:
            i = random.randrange(1, len(self.path) - 2)
            j = random.randrange(i + 2, len(self.path))
            return LocalMove(i, j)
        else:
            return None

    def random_local_moves_wor(self) -> Iterable[LocalMove]:
        for i, j in sample2(len(self.path), len(self.path)):
            if i >= 1 and j >= i + 2:
                yield LocalMove(i, j)

    def heuristic_add_move(self) -> Optional[Component]:
        # Return the closest
        if len(self.path) < self.problem.nnodes:
            best = None
            bestd = None
            u = self.path[-1]
            for v in self.unused:
                d = self.problem.dist[u][v]
                if bestd is None or d < bestd:
                    best = Component(u, v)
                    bestd = d
            return best
        elif len(self.path) == self.problem.nnodes:
            u = self.path[-1]
            return Component(u, self.start)
        return None

    def add(self, component: Component) -> None:
        u, v = component.u, component.v
        self.path.append(v)
        if v != self.start:
            self.unused.remove(v)
        self.used.add(v)
        self.dist += self.problem.dist[u][v]

    def step(self, lmove: LocalMove) -> None:
        i, j = lmove.i, lmove.j
        self.dist -= self.problem.dist[self.path[i - 1]][self.path[i]]
        self.dist -= self.problem.dist[self.path[j - 1]][self.path[j]]
        self.path[i:j] = list(reversed(self.path[i:j]))
        self.dist += self.problem.dist[self.path[i - 1]][self.path[i]]
        self.dist += self.problem.dist[self.path[j - 1]][self.path[j]]
        if self.logger:
            self.logger(self.dist)
        if __debug__:
            dist = sum(
                map(lambda t: self.problem.dist[t[0]]
                    [t[1]], pairwise(self.path))
            )
            assert isclose(dist, self.dist), (dist, self.dist)
            assert self.path[0] == self.start, self.path
            assert self.path[-1] == self.start, self.path

    def objective_incr_local(self, lmove: LocalMove) -> Optional[float]:
        i, j = lmove.i, lmove.j
        ndist = self.dist
        ndist -= self.problem.dist[self.path[i - 1]][self.path[i]]
        ndist -= self.problem.dist[self.path[j - 1]][self.path[j]]
        ndist += self.problem.dist[self.path[i - 1]][self.path[j - 1]]
        ndist += self.problem.dist[self.path[i]][self.path[j]]
        return ndist - self.dist

    def lower_bound_incr_add(self, component: Component) -> Optional[float]:
        if len(self.path) + 1 <= cast(Problem, self.problem).nnodes:
            u, v = component.u, component.v
            d = self.problem.dist[u][v]
            return d
        else:
            return 0

    def perturb(self, ks: int) -> None:
        for _ in range(ks):
            move = self.random_local_move()
            if move is not None:
                self.step(move)

    def components(self) -> Iterable[Component]:
        for i in range(1, len(self.path)):
            yield Component(self.path[i - 1], self.path[i])


@dataclass
class Point:
    x: float
    y: float


if sys.version_info < (3, 9):
    CoordList = Sequence
    DistMatrix = tuple
else:
    CoordList = Sequence[Point]
    DistMatrix = tuple[tuple[float, ...], ...]


def euclidean_dist(a: Point, b: Point) -> float:
    dx = a.x - b.x
    dy = a.y - b.y
    return sqrt(dx * dx + dy * dy)


def distance_matrix(coords: CoordList) -> DistMatrix:
    mat = []
    for a in coords:
        row = []
        for b in coords:
            row.append(euclidean_dist(a, b))
        mat.append(tuple(row))
    return tuple(mat)


class Problem:
    def __init__(self, coords: CoordList) -> None:
        self.nnodes = len(coords)
        self.coords = coords
        self.dist = distance_matrix(coords)

    @classmethod
    def from_textio(cls, f: TextIO) -> Problem:
        n = int(f.readline())
        coords = []
        for _ in range(n):
            x, y = map(float, f.readline().split())
            coords.append(Point(x, y))
        return cls(coords)

    def empty_solution(self, logger: Optional[BasicLogger] = None) -> Solution:
        return Solution(
            self, 0, Path([0]), Used({0}), Unused(
                set(range(1, self.nnodes))), 0, logger
        )

    def empty_solution_with_start(
        self, start: int, logger: Optional[BasicLogger] = None
    ) -> Solution:
        return Solution(
            self,
            start,
            Path([start]),
            Used({start}),
            Unused(set(range(self.nnodes)) - {start}),
            0,
            logger
        )


if __name__ == "__main__":
    from api.solvers import *
    from time import perf_counter
    import argparse
    import sys

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug"],
        default="warning",
    )
    parser.add_argument(
        "--log-file", type=argparse.FileType("w"), default=sys.stderr)
    parser.add_argument(
        "--csearch",
        choices=["beam", "grasp", "greedy", "heuristic", "as", "mmas", "none"],
        default="none",
    )
    parser.add_argument("--cbudget", type=float, default=5.0)
    parser.add_argument(
        "--lsearch", choices=["bi", "fi", "ils", "rls", "sa", "none"], default="none"
    )
    parser.add_argument("--lbudget", type=float, default=5.0)
    parser.add_argument(
        "--input-file", type=argparse.FileType("r"), default=sys.stdin)
    parser.add_argument(
        "--output-file", type=argparse.FileType("w"), default=sys.stdout
    )
    parser.add_argument("--performance-file", type=str, required=False)
    args = parser.parse_args()

    logging.basicConfig(
        stream=args.log_file,
        level=args.log_level.upper(),
        format="%(levelname)s;%(asctime)s;%(message)s",
    )

    if args.performance_file:
        base_logger = BasicLogger(
            f"{args.csearch}_{args.lsearch}", args.input_file.name
        )
    else:
        base_logger = None

    p = Problem.from_textio(args.input_file)
    s: Optional[Solution] = p.empty_solution(base_logger)

    start = perf_counter()


    if s is not None and args.csearch != "none":
        if args.csearch == "beam":
            s = beam_search(s, 10)
        elif args.csearch == "grasp":
            # s = grasp(s, args.cbudget, alpha = 0.01)
            s = grasp(
                s,
                args.cbudget,
                alpha=0.01,
                local_search=lambda s: first_improvement(s, 0.1),
            )
        elif args.csearch == "greedy":
            s = greedy_construction(s)
        elif args.csearch == "heuristic":
            s = heuristic_construction(s)
        elif args.csearch == "as":
            ants = [p.empty_solution_with_start(i, base_logger) for i in range(p.nnodes)]
            # s = aco(ants, 30.0, beta = 5.0, rho = 0.5, tau0 = 1 / 3000.0)
            lbudget = 1.0 / len(ants)
            s = ant_system(
                ants,
                args.cbudget,
                beta=5.0,
                rho=0.5,
                tau0=1 / 3000.0,
                local_search=lambda s: first_improvement(s, lbudget),
            )
        elif args.csearch == "mmas":
            ants = [p.empty_solution_with_start(i, base_logger) for i in range(p.nnodes)]
            # s = mmas(ants, 30.0, beta = 5.0, rho = 0.02, taumax = 1 / 3000.0, ants = 0.1)
            lbudget = 1.0 / len(ants)
            s = mmas(
                ants,
                args.cbudget,
                beta=5.0,
                rho=0.05,
                taumax=1 / 3000.0,
                globalratio=0.1,
                local_search=lambda s: first_improvement(s, lbudget),
            )

    if s is not None:
        if args.lsearch == "bi":
            s = best_improvement(s, args.lbudget)
        elif args.lsearch == "fi":
            s = first_improvement(s, args.lbudget)
        elif args.lsearch == "ils":
            s = ils(s, args.lbudget)
        elif args.lsearch == "rls":
            s = rls(s, args.lbudget)
        elif args.lsearch == "sa":
            s = sa(s, args.lbudget, 10)

    end = perf_counter()
    if s is not None:
        print(s.output(), file=args.output_file)
        if s.objective() is not None:
            logging.info(f"Objective: {s.objective():.3f}")
        else:
            logging.info(f"Objective: None")
    else:
        logging.info(f"Objective: no solution found")
    if base_logger:
        base_logger.write(args.performance_file)
    logging.info(f"Elapsed solving time: {end-start:.4f}")
