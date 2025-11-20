import json
import os
import itertools
from typing import List, Dict, Tuple
from tournament import play
from connect4.policy import Policy
from connect4.utils import find_importable_classes

# Dynamically locate Aha policy from Group A without direct import (spaces in path)
_policies = find_importable_classes("groups", Policy)
Aha = _policies.get("Group A")  # type: ignore
if Aha is None:
    raise RuntimeError("Aha policy (Group A) not found. Ensure Group A/policy.py exists.")

# Factory to create fixed-parameter variant subclasses

def make_variant_class(name: str, num_simulations: int, exploration_weight: float, rollout_depth: int, heuristics_enabled: bool):
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


def build_variants(limit: int | None = 12) -> List[Tuple[str, type]]:
    """Generate variants via filtered grid instead of fixed list.

    Filtering rules:
    - Cost = num_simulations * rollout_depth <= MAX_COST
    - If heuristics_enabled True, skip very high depths (> 25) to avoid redundancy
    - Provide optional 'limit' to cap number of returned variants for quick runs
    """
    MAX_COST = 4500  # tighter to force diversity in sims range 100-200
    # Restrict simulations to desired exploration band 100-200
    num_simulations_list = [100, 120, 140, 160, 180, 200]
    exploration_weights = [0.8, 1.0, 1.2, 1.414]
    rollout_depths = [12, 15, 18, 20, 22, 25]
    heuristic_flags = [False, True]

    raw = itertools.product(num_simulations_list, exploration_weights, rollout_depths, heuristic_flags)
    candidates: List[Dict] = []
    for ns, ew, rd, hf in raw:
        cost = ns * rd
        if cost > MAX_COST:
            continue
        if hf and rd > 25:
            continue
        candidates.append({
            "num_simulations": ns,
            "exploration_weight": ew,
            "rollout_depth": rd,
            "heuristics_enabled": hf,
            "cost": cost
        })

    # Stratified selection: ensure representation across num_simulations values
    candidates.sort(key=lambda c: (c["num_simulations"], c["cost"]))
    if limit is not None:
        stratified = []
        by_ns: Dict[int, List[Dict]] = {}
        for c in candidates:
            by_ns.setdefault(c["num_simulations"], []).append(c)
        # pick up to ceil(limit / len(num_simulations_list)) per simulation bucket
        import math as _math
        per_bucket = max(1, _math.ceil(limit / len(num_simulations_list)))
        for ns in sorted(by_ns.keys()):
            picked = by_ns[ns][:per_bucket]
            stratified.extend(picked)
            if len(stratified) >= limit:
                break
        candidates = stratified[:limit]

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

    # Persist candidate list for later analysis
    os.makedirs("versus", exist_ok=True)
    with open("versus/grid_candidates.json", "w") as f:
        json.dump(candidates, f, indent=2)
    return variants


def round_robin_variants(players: List[Tuple[str, type]], best_of: int = 3, first_player_distribution: float = 0.5, seed: int = 911):
    os.makedirs("versus", exist_ok=True)
    names = [p[0] for p in players]
    # Initialize result matrices
    matrix: Dict[str, Dict[str, Dict[str, float]]] = {n: {} for n in names}
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

    summary = {
        "variants": names,
        "matrix": matrix,
        "scores": scores,
        "ranking": sorted(scores.items(), key=lambda x: x[1], reverse=True),
    }
    with open("versus/variants_summary.json", "w") as f:
        json.dump(summary, f, indent=4)
    return summary


if __name__ == "__main__":
    variants = build_variants()
    # Load candidate configs for printing (already persisted by build_variants)
    try:
        with open("versus/grid_candidates.json", "r") as f:
            candidate_data = json.load(f)
    except FileNotFoundError:
        candidate_data = []

    print("\nCANDIDATOS A EVALUAR (antes de competir):")
    if candidate_data:
        print(f"Total: {len(candidate_data)}")
        print("sims  expo   depth  heur  cost")
        for c in candidate_data:
            print(f"{c['num_simulations']:>4}  {c['exploration_weight']:<5}  {c['rollout_depth']:>5}  {'on' if c['heuristics_enabled'] else 'off':>4}  {c['cost']:>5}")
    else:
        print("(No candidate data available)")

    summary = round_robin_variants(variants)
    print("Ranking:")
    for rank, (name, score) in enumerate(summary["ranking"], start=1):
        print(f"{rank}. {name}: {score:.2f} pts")
