"""
API das funcionalidades do sistema
"""

import copy
import heapq
import itertools
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Iterable

import numpy as np
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.models import DiscreteBayesianNetwork


class Fever(StrEnum):
    Normal = auto()
    High = auto()


class Saturation(StrEnum):
    Normal = auto()
    Reduced = auto()
    Critical = auto()


class Pressure(StrEnum):
    Normal = auto()
    Low = auto()


class Frequency(StrEnum):
    Normal = auto()
    High = auto()


class Pain(StrEnum):
    Low = auto()
    Medium = auto()
    High = auto()


class Age(StrEnum):
    Adult = auto()
    Elderly = auto()


class Illness(StrEnum):
    No = auto()
    Yes = auto()


class Severity(StrEnum):
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

        self.PROB = {
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
                for k, v in self.PROB.items()
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

    def severity_infer(self, report: Report):
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
        )

    def severity(self, report: Report):
        return self.severity_infer(report).get_value(Severity=Severity.High.name)

    def risk(self, report: Report):
        return self.severity(report) * report.wait_time

    def risk_acc(self, reports: Iterable[Report]):
        s = 0
        for r in reports:
            s += self.risk(r)
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
        self.state: tuple[int, ...] = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.level = 0 if not parent else parent.level + 1

    def __lt__(self, other):
        return self.path_cost < other.path_cost

    def solution(self):
        actions = []
        curr = self
        while curr.parent is not None:
            actions.append(curr.action)
            curr = curr.parent
        return actions[::-1]

    def is_goal(self):
        return len(self.state) == 0


def order_a_star(q: list[Report], consultation_time, sn: SeverityBayesianNetwork):
    q = copy.deepcopy(q)

    def expand(node: _SearchNode):
        s = node.state
        for i in s:
            s1 = tuple(j for j in s if j != i)
            cost = node.path_cost
            wait_time = consultation_time * node.level
            for k in s1:
                cost += sn.severity(q[k]) * (q[k].wait_time + wait_time)
            yield _SearchNode(state=s1, parent=node, action=i, path_cost=cost)

    def h_func(node: _SearchNode):
        wait_time = consultation_time * node.level
        cost = 0
        for k in node.state:
            cost += sn.severity(q[k]) * (q[k].wait_time + wait_time)
        return cost

    node = _SearchNode(tuple(range(len(q))))
    frontier = []
    heapq.heappush(frontier, (node.path_cost + h_func(node), node))
    reached = {node.state: node}

    while frontier:
        priority, node = heapq.heappop(frontier)
        # if priority > node.path_cost + h_func(node):
        #     continue
        if node.is_goal():
            res = [q[i] for i in node.solution()]
            for i, r in enumerate(res):
                r.wait_time += i * consultation_time
            return res
        for child in expand(node):
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
