"""
Testes de caso e comparações entre estratégias.
"""

import copy
import random

import triage


class TestTriageManager(triage.TriageManager):
    def __init__(self, consultation_time):
        super().__init__(consultation_time)

    def order_fifo(self, q: list[triage.Report]):
        q = copy.deepcopy(q)
        order = []
        while q:
            p = q.pop(0)
            order.append(p)
            for p in q:
                p.wait_time += self.consultation_time
        return order

    def order_greedy(self, q: list[triage.Report]):
        q = copy.deepcopy(q)
        order = []
        while q:
            p = q.pop(q.index(max(q, key=lambda r: self.sn.severity(r))))
            order.append(p)
            for p in q:
                p.wait_time += self.consultation_time
        return order


def make_queue(size: int):
    random.seed(42)
    q = []
    for _ in range(size):
        q.append(
            triage.Report(
                fever=random.choice(list(triage.Fever)),
                saturation=random.choice(list(triage.Saturation)),
                pressure=random.choice(list(triage.Pressure)),
                frequency=random.choice(list(triage.Frequency)),
                pain=random.choice(list(triage.Pain)),
                age=random.choice(list(triage.Age)),
                illness=random.choice(list(triage.Illness)),
                wait_time=random.randint(0, 40),
            )
        )
    return q


def main():
    t = TestTriageManager(20)
    small_q = make_queue(8)
    medium_q = make_queue(16)

    print()
    print("Teste usando fila pequena:")
    print("  FIFO:   ", t.sn.risk_acc(t.order_fifo(small_q)))
    print("  Greedy: ", t.sn.risk_acc(t.order_greedy(small_q)))
    print("  A*:     ", t.sn.risk_acc(t.order(small_q)))

    print()
    print("Teste usando fila média:")
    print("  FIFO:   ", t.sn.risk_acc(t.order_fifo(medium_q)))
    print("  Greedy: ", t.sn.risk_acc(t.order_greedy(medium_q)))
    print("  A*:     ", t.sn.risk_acc(t.order(medium_q)))


if __name__ == "__main__":
    main()
