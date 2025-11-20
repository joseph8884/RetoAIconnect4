import json
import os
import random
import argparse
from typing import List, Dict, Tuple
from tournament import play
from connect4.policy import Policy
from connect4.utils import find_importable_classes

# Localiza dinámicamente la policy Aha del Grupo A
_policies = find_importable_classes("groups", Policy)
Aha = _policies.get("Group A")  # type: ignore
if Aha is None:
    raise RuntimeError("Aha policy (Group A) not found. Ensure Group A/policy.py exists.")

def make_variant_class(name: str, num_simulations: int, exploration_weight: float, rollout_depth: int, heuristics_enabled: bool):
    """Crea una subclase de Aha con hiperparámetros fijos."""
    class VariantAha(Aha):  # type: ignore
        def __init__(self):
            super().__init__(
                num_simulations=num_simulations,
                exploration_weight=exploration_weight,
                rollout_depth=rollout_depth,
                heuristics_enabled=heuristics_enabled,
            )
    VariantAha.__name__ = name
    return VariantAha


def build_random_variants(samples: int, max_cost: int, seed: int) -> List[Tuple[str, type]]:
    """Genera 'samples' variantes aleatorias dentro de un costo máximo.

    Costo = num_simulations * rollout_depth. Evita duplicados exactos.
    """
    rng = random.Random(seed)
    exploration_weights = [0.8, 1.0, 1.2, 1.414]
    # Use discrete candidate sets (previous grid lists) instead of full ranges
    sims_candidates = [100, 120, 140, 160, 180, 200]
    depth_candidates = [12, 15, 18, 20, 22, 25]

    seen: set[tuple] = set()
    candidates: List[Dict] = []
    attempts = 0
    max_attempts = samples * 10
    while len(candidates) < samples and attempts < max_attempts:
        attempts += 1
        ns = rng.choice(sims_candidates)
        rd = rng.choice(depth_candidates)
        ew = rng.choice(exploration_weights)
        hf = rng.choice([False, True])
        cost = ns * rd
        if cost > max_cost:
            continue
        key = (ns, ew, rd, hf)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({
            "num_simulations": ns,
            "exploration_weight": ew,
            "rollout_depth": rd,
            "heuristics_enabled": hf,
            "cost": cost
        })

    variants: List[Tuple[str, type]] = []
    for cfg in candidates:
        name = (
            f"Aha_s{cfg['num_simulations']}_e{cfg['exploration_weight']}_d{cfg['rollout_depth']}"
            f"_h{'on' if cfg['heuristics_enabled'] else 'off'}"
        )
        cls = make_variant_class(name,
                                 cfg["num_simulations"],
                                 cfg["exploration_weight"],
                                 cfg["rollout_depth"],
                                 cfg["heuristics_enabled"])
        variants.append((name, cls))

    os.makedirs("versus", exist_ok=True)
    with open("versus/random_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
    return variants


def round_robin_variants(players: List[Tuple[str, type]], best_of: int = 3, first_player_distribution: float = 0.5, seed: int = 911):
    """Juega todos contra todos y retorna resumen con ranking.

    Puntuación: victoria=1, empate=0.5.
    """
    os.makedirs("versus", exist_ok=True)
    names = [p[0] for p in players]
    # Initialize result matrices
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {n: {} for n in names}  # se deja por si se quiere analizar detalle
    scores: Dict[str, float] = {n: 0.0 for n in names}

    for i in range(len(players)):
        for j in range(i + 1, len(players)):
            a = players[i]
            b = players[j]
            print(f"Playing {a[0]} vs {b[0]} ...")
            winner = play(a, b, best_of=best_of, first_player_distribution=first_player_distribution, seed=seed)
            a_name, _ = a
            b_name, _ = b
            # Load match data written by play()
            filename = f"versus/match_{a_name}_vs_{b_name}.json"
            with open(filename, "r") as f:
                match = json.load(f)
            a_wins = match["player_a_wins"]
            b_wins = match["player_b_wins"]
            draws = match["draws"]
            total = a_wins + b_wins + draws if (a_wins + b_wins + draws) > 0 else 1
            a_points = a_wins + 0.5 * draws
            b_points = b_wins + 0.5 * draws
            # Store per-pair stats
            matrix[a_name][b_name] = {
                "a_wins": a_wins,
                "b_wins": b_wins,
                "draws": draws,
                "a_win_rate": a_wins / total,
                "b_win_rate": b_wins / total,
                "draw_rate": draws / total,
                "a_points": a_points,
                "b_points": b_points,
            }
            matrix[b_name][a_name] = {
                "a_wins": b_wins,
                "b_wins": a_wins,
                "draws": draws,
                "a_win_rate": b_wins / total,
                "b_win_rate": a_wins / total,
                "draw_rate": draws / total,
                "a_points": b_points,
                "b_points": a_points,
            }
            scores[a_name] += a_points
            scores[b_name] += b_points

    ranking = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    summary = {"variants": names, "scores": scores, "ranking": ranking}
    with open("versus/variants_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    return summary


def evaluate_multiple_runs(runs: int, samples: int, max_cost: int, seed: int, match_best_of: int, final_best_of: int, first_player: float):
    """Ejecuta varios torneos aleatorios y luego un playoff entre ganadores únicos."""
    os.makedirs("versus", exist_ok=True)
    run_winners: List[Dict] = []
    winner_configs: List[Dict] = []
    base_seed = seed

    for r in range(1, runs + 1):
        run_seed = base_seed + r
        print(f"\n=== RUN {r}/{runs} (seed={run_seed}) ===")
        variants = build_random_variants(samples, max_cost, run_seed)
        # Carga configuraciones usadas en este run
        try:
            with open("versus/random_candidates.json", "r") as f:
                candidate_data = json.load(f)
        except FileNotFoundError:
            candidate_data = []
        summary = round_robin_variants(variants, best_of=match_best_of, first_player_distribution=first_player, seed=run_seed)
        top_name, top_score = summary["ranking"][0]
        # Busca config del ganador
        matched_cfg = next((c for c in candidate_data if (
            f"Aha_s{c['num_simulations']}_e{c['exploration_weight']}_d{c['rollout_depth']}_h" + ("on" if c['heuristics_enabled'] else "off")
        ) == top_name), {"error": "config_not_found", "name": top_name})
        run_winners.append({"run": r, "winner": top_name, "score": top_score})
        winner_configs.append(matched_cfg)

    # Deduplicate winner configs by their identifying tuple
    unique_map: Dict[tuple, Dict] = {}
    for c in winner_configs:
        if "error" in c:
            continue
        key = (c["num_simulations"], c["exploration_weight"], c["rollout_depth"], c["heuristics_enabled"])
        if key not in unique_map:
            unique_map[key] = c

    final_candidates = list(unique_map.values())
    print(f"\n=== PLAYOFF FINAL con {len(final_candidates)} ganadores únicos ===")
    final_variants: List[Tuple[str, type]] = []
    for cfg in final_candidates:
        name = (
            f"Aha_s{cfg['num_simulations']}_e{cfg['exploration_weight']}_d{cfg['rollout_depth']}"
            f"_h{'on' if cfg['heuristics_enabled'] else 'off'}"
        )
        cls = make_variant_class(name, cfg["num_simulations"], cfg["exploration_weight"], cfg["rollout_depth"], cfg["heuristics_enabled"])
        final_variants.append((name, cls))

    playoff_summary = round_robin_variants(final_variants, best_of=final_best_of, first_player_distribution=first_player, seed=base_seed + 999)
    champion_name, champion_score = playoff_summary["ranking"][0]
    champion_cfg = None
    for c in final_candidates:
        nm = (
            f"Aha_s{c['num_simulations']}_e{c['exploration_weight']}_d{c['rollout_depth']}_h" +
            ("on" if c['heuristics_enabled'] else "off")
        )
        if nm == champion_name:
            champion_cfg = c
            break

    result = {
        "runs": runs,
        "run_winners": run_winners,
        "unique_winner_configs": final_candidates,
        "playoff_ranking": playoff_summary["ranking"],
        "champion": {
            "name": champion_name,
            "score": champion_score,
            "config": champion_cfg,
        },
    }
    with open("versus/run_winners.json", "w") as f:
        json.dump(run_winners, f, indent=2)
    with open("versus/final_best.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nFINAL CHAMPION: {champion_name} (score={champion_score:.2f})")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evalúa variantes aleatorias de Aha y opcionalmente hace múltiples rondas + playoff.")
    parser.add_argument("--samples", type=int, default=8, help="Number of random variants to generate")
    parser.add_argument("--best-of", type=int, default=3, help="Games per match (odd number recommended)")
    parser.add_argument("--seed", type=int, default=911, help="Random seed")
    parser.add_argument("--max-cost", type=int, default=3600, help="Maximum cost (num_simulations*rollout_depth)")
    parser.add_argument("--first-player", type=float, default=0.5, help="Probability that listed player starts")
    parser.add_argument("--runs", type=int, default=1, help="Number of independent random tournaments to run before playoff")
    parser.add_argument("--final-best-of", type=int, default=5, help="Games per match in final playoff")
    args = parser.parse_args()

    if args.runs <= 1:
        variants = build_random_variants(args.samples, args.max_cost, args.seed)
        try:
            with open("versus/random_candidates.json", "r") as f:
                candidate_data = json.load(f)
        except FileNotFoundError:
            candidate_data = []
        print("\nCANDIDATOS ALEATORIOS:")
        if candidate_data:
            print(f"Total: {len(candidate_data)}")
            print("sims  expo   depth  heur  cost")
            for c in candidate_data:
                print(f"{c['num_simulations']:>4}  {c['exploration_weight']:<5}  {c['rollout_depth']:>5}  {'on' if c['heuristics_enabled'] else 'off':>4}  {c['cost']:>5}")
        else:
            print("(No candidate data)")
        summary = round_robin_variants(variants, best_of=args.best_of, first_player_distribution=args.first_player, seed=args.seed)
        print("\nRanking:")
        for rank, (name, score) in enumerate(summary["ranking"], start=1):
            print(f"{rank}. {name}: {score:.2f} pts")
    else:
        evaluate_multiple_runs(
            runs=args.runs,
            samples=args.samples,
            max_cost=args.max_cost,
            seed=args.seed,
            match_best_of=args.best_of,
            final_best_of=args.final_best_of,
            first_player=args.first_player,
        )
