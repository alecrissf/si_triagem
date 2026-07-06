"""
API das funcionalidades do sistema
"""

import heapq
import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import Iterable

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


class Fever(Enum):
    Normal = auto()
    High = auto()


class Saturation(Enum):
    Normal = auto()
    Reduced = auto()
    Critical = auto()


class Pressure(Enum):
    Normal = auto()
    Low = auto()


class Frequency(Enum):
    Normal = auto()
    High = auto()


class Pain(Enum):
    Low = auto()
    Medium = auto()
    High = auto()


class Age(Enum):
    Adult = auto()
    Elderly = auto()


class Illness(Enum):
    No = auto()
    Yes = auto()


class Severity(Enum):
    Low = auto()
    Medium = auto()
    High = auto()


@dataclass(unsafe_hash=True)
class Report:
    fever: Fever
    saturation: Saturation
    pressure: Pressure
    frequency: Frequency
    pain: Pain
    age: Age
    illness: Illness
    wait_time: float = 0


class SeverityBayesianNetwork:
    def __init__(self):
        self.STATES = {
            e.__name__: _enum_names(e)
            for e in [
                Fever,
                Saturation,
                Pressure,
                Frequency,
                Pain,
                Age,
                Illness,
                Severity,
            ]
        }

        self.nw = DiscreteBayesianNetwork(
            list(
                itertools.product(
                    [
                        Fever.__name__,
                        Saturation.__name__,
                        Pressure.__name__,
                        Frequency.__name__,
                        Pain.__name__,
                        Age.__name__,
                        Illness.__name__,
                    ],
                    [Severity.__name__],
                )
            )
        )

        PROB = {
            Fever.__name__: [0.8, 0.2],
            Saturation.__name__: [0.10, 0.35, 0.55],
            Pressure.__name__: [0.85, 0.15],
            Frequency.__name__: [0.75, 0.25],
            Pain.__name__: [0.45, 0.35, 0.20],
            Age.__name__: [0.70, 0.30],
            Illness.__name__: [0.65, 0.35],
        }

        self.nw.add_cpds(
            *[
                TabularCPD(
                    variable=k,
                    variable_card=len(v),
                    values=[[p] for p in v],
                    state_names={k: self.STATES[k]},
                )
                for k, v in PROB.items()
            ]
        )

        combinations = list(
            itertools.product(
                self.STATES[Fever.__name__],
                self.STATES[Saturation.__name__],
                self.STATES[Pressure.__name__],
                self.STATES[Frequency.__name__],
                self.STATES[Pain.__name__],
                self.STATES[Age.__name__],
                self.STATES[Illness.__name__],
            )
        )
        self.nw.add_cpds(
            TabularCPD(
                variable=Severity.__name__,
                variable_card=len(Severity),
                values=np.array([_severity_prob(*c) for c in combinations]).T,
                evidence=[
                    Fever.__name__,
                    Saturation.__name__,
                    Pressure.__name__,
                    Frequency.__name__,
                    Pain.__name__,
                    Age.__name__,
                    Illness.__name__,
                ],
                evidence_card=[
                    len(Fever),
                    len(Saturation),
                    len(Pressure),
                    len(Frequency),
                    len(Pain),
                    len(Age),
                    len(Illness),
                ],
                state_names=self.STATES,
            )
        )

        self.nw.check_model()
        self.infer = VariableElimination(self.nw)

    def severity(self, report: Report):
        return self.infer.query(
            [Severity.__name__],
            {
                Fever.__name__: report.fever.name,
                Saturation.__name__: report.saturation.name,
                Pressure.__name__: report.pressure.name,
                Frequency.__name__: report.frequency.name,
                Pain.__name__: report.pain.name,
                Age.__name__: report.age.name,
                Illness.__name__: report.illness.name,
            },  # ty:ignore[invalid-argument-type]
        ).get_value(Severity=Severity.High.name)

    def risk(self, report: Report):
        return self.severity(report) * report.wait_time

    def risk_acc(self, reports: Iterable[Report]):
        s = 0
        for r in reports:
            s += self.risk(r)
        return s

    def _severity_tup(self, report_tup: tuple):
        return self.infer.query(
            [Severity.__name__],
            {
                Fever.__name__: report_tup[0].name,
                Saturation.__name__: report_tup[1].name,
                Pressure.__name__: report_tup[2].name,
                Frequency.__name__: report_tup[3].name,
                Pain.__name__: report_tup[4].name,
                Age.__name__: report_tup[5].name,
                Illness.__name__: report_tup[6].name,
            },
        ).get_value(Severity=Severity.High.name)

    def _risk_tup(self, report_tup: tuple):
        return self._severity_tup(report_tup) * report_tup[-1]

    def _risk_acc_tup(self, reports: tuple[tuple, ...]):
        s = 0
        for r in reports:
            s += self._risk_tup(r)
        return s


def _enum_names(e):
    return [m.name for m in e]


def _severity_prob(fever, saturation, pressure, frequency, pain, age, illness):
    WEIGHTS = {
        Fever.__name__: {"Normal": 0, "High": 1},
        Saturation.__name__: {"Normal": 0, "Reduced": 3, "Critical": 6},
        Pressure.__name__: {"Normal": 0, "Low": 2},
        Frequency.__name__: {"Normal": 0, "High": 2},
        Pain.__name__: {"Low": 0, "Medium": 1, "High": 2},
        Age.__name__: {"Adult": 0, "Elderly": 2},
        Illness.__name__: {"No": 0, "Yes": 2},
    }

    severity = 0

    severity += WEIGHTS[Fever.__name__][fever]
    severity += WEIGHTS[Saturation.__name__][saturation]
    severity += WEIGHTS[Pressure.__name__][pressure]
    severity += WEIGHTS[Frequency.__name__][frequency]
    severity += WEIGHTS[Pain.__name__][pain]
    severity += WEIGHTS[Age.__name__][age]
    severity += WEIGHTS[Illness.__name__][illness]

    if saturation == "Critical" and pressure == "Low":
        severity += 3

    if saturation == "Reduced" and pressure == "Low":
        severity += 2

    if pressure == "Low" and frequency == "High":
        severity += 1

    if frequency == "High" and pain == "High":
        severity += 2

    if age == "Elderly" and illness == "Yes":
        severity += 1

    if severity <= 2:
        return [0.96, 0.03, 0.01]
    elif severity <= 5:
        return [0.70, 0.25, 0.05]
    elif severity <= 8:
        return [0.35, 0.45, 0.20]
    elif severity <= 11:
        return [0.12, 0.33, 0.55]
    elif severity <= 14:
        return [0.05, 0.20, 0.75]
    else:
        return [0.01, 0.09, 0.90]


class _SearchNode:
    def __init__(self, state, parent=None, action=None, path_cost=0):
        self.state: tuple[tuple, ...] = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def solution(self):
        actions = []
        curr = self
        while curr.parent is not None:
            actions.append(curr.action)
            curr = curr.parent
        return actions[::-1]


def expand(node: _SearchNode, consultation_time, sn: SeverityBayesianNetwork):
    s = node.state
    for i in range(len(s)):
        s1 = tuple(
            (*r[:7], r[7] + consultation_time) for j, r in enumerate(s) if j != i
        )

        cost = node.path_cost + sn._risk_acc_tup(s1)
        yield _SearchNode(state=s1, parent=node, action=i, path_cost=cost)


def order_a_star(q: list[Report], consultation_time, sn: SeverityBayesianNetwork):
    init_q = tuple(
        (
            r.fever,
            r.saturation,
            r.pressure,
            r.frequency,
            r.pain,
            r.age,
            r.illness,
            r.wait_time,
        )
        for r in sorted(q, key=lambda r: sn.risk(r), reverse=True)
    )
    node = _SearchNode(init_q)

    def h_func(node: _SearchNode):
        return sn._risk_acc_tup(node.state)

    frontier = []
    heapq.heappush(frontier, (node.path_cost + h_func(node), node))
    reached = {init_q: node}

    while frontier:
        priority, node = heapq.heappop(frontier)
        if priority > node.path_cost + h_func(node):
            continue
        if len(node.state) == 0:
            return [
                Report(
                    *init_q[i][:7],
                    wait_time=init_q[i][-1] + i * consultation_time,
                )
                for i in node.solution()
            ]
        for child in expand(node, consultation_time, sn):
            s = child.state
            if s not in reached:
                reached[s] = child
                heapq.heappush(frontier, (child.path_cost + h_func(child), child))

    return []


class TriageManager:
    def __init__(self, consultation_time):
        self.consultation_time = consultation_time
        self.sn = SeverityBayesianNetwork()

    def order(self, q: list[Report]):
        return order_a_star(q, self.consultation_time, self.sn)
