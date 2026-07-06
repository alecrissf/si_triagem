"""
Testes de caso e comparações entre estratégias.
"""

import copy
import csv
import pathlib
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


def choose(e, prob_dict):
    return random.choices(list(e), weights=prob_dict[e.__name__])[0]


def make_queue(size: int, sn: triage.SeverityBayesianNetwork):
    random.seed(42)
    q = []
    for _ in range(size):
        q.append(
            triage.Report(
                fever=choose(triage.Fever, sn.PROB),
                saturation=choose(triage.Saturation, sn.PROB),
                pressure=choose(triage.Pressure, sn.PROB),
                frequency=choose(triage.Frequency, sn.PROB),
                pain=choose(triage.Pain, sn.PROB),
                age=choose(triage.Age, sn.PROB),
                illness=choose(triage.Illness, sn.PROB),
                wait_time=random.randint(0, 40),
            )
        )
    return q


def write_q_to_csv(q: list[triage.Report], filename: pathlib.Path):
    with open(filename, "w", newline="") as file:
        w = csv.writer(file)
        w.writerow(
            [
                "fever",
                "saturation",
                "pressure",
                "frequency",
                "pain",
                "age",
                "illness",
                "wait_time",
            ]
        )
        for r in q:
            w.writerow(
                [
                    r.fever,
                    r.saturation,
                    r.pressure,
                    r.frequency,
                    r.pain,
                    r.age,
                    r.illness,
                    r.wait_time,
                ]
            )

    print(f"Escrito arquivo {filename}")


def test_queue(t: TestTriageManager, size: int, out_dir: pathlib.Path):
    q = make_queue(size, t.sn)

    print(f"\nTestando estratégias com fila de tamanho {size}:")
    fifo = t.order_fifo(q)
    print("  FIFO:   ", t.sn.risk_acc(fifo))
    greedy = t.order_greedy(q)
    print("  Guloso: ", t.sn.risk_acc(greedy))
    a_star = t.order(q)
    print("  A*:     ", t.sn.risk_acc(a_star))

    print()
    write_q_to_csv(q, out_dir / f"q_{size:02}_original.csv")
    write_q_to_csv(fifo, out_dir / f"q_{size:02}_fifo.csv")
    write_q_to_csv(greedy, out_dir / f"q_{size:02}_greedy.csv")
    write_q_to_csv(a_star, out_dir / f"q_{size:02}_a_star.csv")


def main():
    out = pathlib.Path.cwd() / "out"
    out.mkdir(parents=True, exist_ok=True)

    t = TestTriageManager(20)

    print(f"Tempo de cada consulta: {t.consultation_time}")

    test_queue(t, 8, out)
    test_queue(t, 16, out)


if __name__ == "__main__":
    main()
