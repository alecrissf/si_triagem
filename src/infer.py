"""
Interface de linha de comando para inferir a probabilidade de risco para um paciente
"""

import typer
from rich import print

import triage

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    fever: triage.Fever,
    saturation: triage.Saturation,
    pressure: triage.Pressure,
    frequency: triage.Frequency,
    pain: triage.Pain,
    age: triage.Age,
    illness: triage.Illness,
    wait_time: float,
):
    s = triage.SeverityBayesianNetwork()
    r = triage.Report(
        fever,
        saturation,
        pressure,
        frequency,
        pain,
        age,
        illness,
        wait_time,
    )
    print(s.severity_infer(r))


if __name__ == "__main__":
    app()
