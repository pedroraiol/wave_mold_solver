# main.py
import argparse
import json
import sys

from core.models import Layer, Problem
from core.postprocess import save_mode_plot
from solvers import BaseSolver
from solvers.fem import FEMSolver
from solvers.tmm import TMMSolver


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wave Mold Solver - Mode Analysis (TMM/FEM)")
    parser.add_argument("-c", "--config", default="config.json", help="Caminho do arquivo de configuração JSON")
    parser.add_argument("--method", choices=["TMM", "FEM"], help="Sobrescreve o método numérico do config")
    parser.add_argument("--wavelength", type=float, help="Sobrescreve o comprimento de onda (nm)")
    parser.add_argument("--polarization", choices=["TE", "TM"], help="Sobrescreve a polarização")
    parser.add_argument("--out-dir", default="outputs", help="Diretório de saída para os PNGs de campo")
    return parser.parse_args()


def load_problem(path: str) -> Problem:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    layers = [Layer(**layer) for layer in cfg["layers"]]

    return Problem(
        wavelength_nm=cfg["wavelength_nm"],
        polarization=cfg["polarization"],
        method=cfg["method"].upper(),
        layers=layers,
    )


def get_solver(method: str) -> BaseSolver:
    if method == "TMM":
        return TMMSolver()
    if method == "FEM":
        return FEMSolver()
    raise ValueError(f"Método inválido: {method}")


def prompt_choice(label: str, choices: list[str], default: str) -> str:
    raw = input(f"{label} [{'/'.join(choices)}] (padrão: {default}): ").strip().upper()
    if not raw:
        return default
    while raw not in choices:
        raw = input(f"Valor inválido. Escolha entre {choices}: ").strip().upper()
    return raw


def prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} (padrão: {default}): ").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        print("Valor inválido, mantendo o padrão.")
        return default


def run_interactive(problem: Problem, out_dir: str) -> tuple[Problem, str]:
    print("Nenhum argumento passado — modo interativo (Enter mantém o valor do config.json).")
    problem.method = prompt_choice("Método numérico", ["TMM", "FEM"], problem.method)
    problem.polarization = prompt_choice("Polarização", ["TE", "TM"], problem.polarization)
    problem.wavelength_nm = prompt_float("Comprimento de onda (nm)", problem.wavelength_nm)
    raw_out = input(f"Diretório de saída (padrão: {out_dir}): ").strip()
    if raw_out:
        out_dir = raw_out
    return problem, out_dir


def main():
    args = parse_args()
    problem = load_problem(args.config)

    if args.method is not None:
        problem.method = args.method
    if args.wavelength is not None:
        problem.wavelength_nm = args.wavelength
    if args.polarization is not None:
        problem.polarization = args.polarization

    out_dir = args.out_dir
    if len(sys.argv) <= 1:
        problem, out_dir = run_interactive(problem, out_dir)

    solver = get_solver(problem.method)
    results = solver.solve(problem)

    if not results:
        print("Nenhum modo guiado encontrado para essa estrutura.")
        return

    for result in results:
        print(f"Modo {result.mode_index}: neff = {result.neff:.6f}, beta = {result.beta:.6e} rad/m")

    for path in save_mode_plot(problem, results, out_dir=out_dir):
        print(f"Perfil de campo salvo em: {path}")


if __name__ == "__main__":
    main()
