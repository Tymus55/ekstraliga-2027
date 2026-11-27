#!/usr/bin/env python3
"""
Monte Carlo simulation of the 2027 PGE Ekstraliga season (10 000 iterations).

Input : sklady_meczowe_prognoza_2027.xlsx  (sheet 'Szczegóły')
Output: wyniki_symulacji_2027.csv
        prognoza_2027.png
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
BASE = Path(__file__).resolve().parent
XLSX = BASE / "sklady_meczowe_prognoza_2027.xlsx"
OUT_CSV = BASE / "wyniki_symulacji_2027.csv"
OUT_PNG = BASE / "prognoza_2027.png"

N_SEASONS = 10_000
RNG_SEED = 2027

W26, W25, K_PRIOR = 1.0, 0.65, 25.0
TARGET_AVG = 1.50  # scale R_i so league mean maps to ~1.50 (no-bonus scale)
HTA_DEFAULT = 0.20
TRACK_NOISE_DEFAULT = 1.00
TRACK_NOISE_BYD = 1.10

MU_BY_POS = {
    "Senior 1": 1.80,
    "Senior 2": 1.80,
    "Senior 3": 1.80,
    "Senior 4": 1.80,
    "U24 / Kevlar": 1.50,
    "Junior 1": 1.15,
    "Junior 2": 1.15,
    "Junior rez.": 0.70,
}

# Start numbers: home 9-15, away 1-7
# Slot order in lineup arrays: S1,S2,S3,S4,U24,J1,J2  -> indices 0..6
HOME_NUMS = np.array([9, 10, 11, 12, 13, 14, 15], dtype=np.int16)
AWAY_NUMS = np.array([1, 2, 3, 4, 5, 6, 7], dtype=np.int16)
NUM_TO_SLOT = {int(n): i for i, n in enumerate(HOME_NUMS)}
NUM_TO_SLOT.update({int(n): i for i, n in enumerate(AWAY_NUMS)})

# Official-style PGE Ekstraliga fixed heats 1–13 (rider numbers)
HEATS_1_13 = np.array(
    [
        [9, 1, 10, 2],
        [7, 15, 6, 14],
        [12, 4, 13, 5],
        [14, 10, 15, 11],
        [1, 12, 2, 13],
        [5, 9, 3, 11],
        [7, 10, 5, 9],
        [3, 13, 4, 14],
        [9, 2, 10, 1],
        [6, 11, 7, 12],
        [4, 9, 5, 10],
        [2, 15, 3, 14],
        [12, 3, 11, 4],
    ],
    dtype=np.int16,
)

TEAM_ORDER = ["TOR", "WRO", "LUB", "ZIE", "BYD", "GOR", "GRU", "LES"]


# ---------------------------------------------------------------------------
# 1. Load & recompute Empirical-Bayes ratings
# ---------------------------------------------------------------------------
def load_lineups(path: Path) -> tuple[dict[str, str], np.ndarray, list[str]]:
    """Return team_names, theta_base[8,7], codes list."""
    df = pd.read_excel(path, sheet_name="Szczegóły")
    df["H25"] = pd.to_numeric(df["H25"], errors="coerce").fillna(0.0)
    df["x25"] = pd.to_numeric(df["x25"], errors="coerce").fillna(0.0)
    df["H26"] = pd.to_numeric(df["H26"], errors="coerce").fillna(0.0)
    df["x26"] = pd.to_numeric(df["x26"], errors="coerce").fillna(0.0)

    # Empirical Bayes with position-specific prior
    mu = df["Pozycja"].map(MU_BY_POS).astype(float)
    num = W26 * df["H26"] * df["x26"] + W25 * df["H25"] * df["x25"] + K_PRIOR * mu
    den = W26 * df["H26"] + W25 * df["H25"] + K_PRIOR
    df["R_i"] = num / den

    # GRU kevlar from juniors with fatigue penalty
    for code in df.loc[df["Kod"] == "GRU", "Kod"].unique():
        j1 = df.loc[(df["Kod"] == code) & (df["Pozycja"] == "Junior 1"), "R_i"].iloc[0]
        j2 = df.loc[(df["Kod"] == code) & (df["Pozycja"] == "Junior 2"), "R_i"].iloc[0]
        kev = 0.90 * 0.5 * (j1 + j2)
        mask = (df["Kod"] == code) & (df["Pozycja"] == "U24 / Kevlar")
        df.loc[mask, "R_i"] = kev

    starters = df[df["W średniej składu"] == "TAK"].copy()
    assert len(starters) == 56, f"expected 56 starters, got {len(starters)}"

    league_avg = float(starters["R_i"].mean())
    starters["R_raw"] = np.clip(starters["R_i"] * (TARGET_AVG / league_avg), 0.05, 2.95)
    starters["theta"] = np.log(starters["R_raw"] / (3.0 - starters["R_raw"]))

    team_names = (
        starters.drop_duplicates("Kod")
        .set_index("Kod")["Drużyna"]
        .reindex(TEAM_ORDER)
        .to_dict()
    )

    pos_order = [
        "Senior 1",
        "Senior 2",
        "Senior 3",
        "Senior 4",
        "U24 / Kevlar",
        "Junior 1",
        "Junior 2",
    ]
    theta = np.zeros((8, 7), dtype=np.float64)
    r_raw = np.zeros((8, 7), dtype=np.float64)
    riders = [[""] * 7 for _ in range(8)]

    for ti, code in enumerate(TEAM_ORDER):
        sub = starters[starters["Kod"] == code]
        for si, pos in enumerate(pos_order):
            row = sub[sub["Pozycja"] == pos].iloc[0]
            theta[ti, si] = float(row["theta"])
            r_raw[ti, si] = float(row["R_raw"])
            riders[ti][si] = str(row["Zawodnik"])

    print("=== Ratingi (Empirical Bayes → R_raw → theta) ===")
    print(f"league_avg(R_i)={league_avg:.3f}  scale→{TARGET_AVG}")
    for ti, code in enumerate(TEAM_ORDER):
        print(f"\n{code} — {team_names[code]}")
        for si, pos in enumerate(pos_order):
            print(
                f"  {pos:12s} {riders[ti][si][:32]:32s} "
                f"R_raw={r_raw[ti, si]:.3f}  theta={theta[ti, si]:+.3f}"
            )

    return team_names, theta, TEAM_ORDER


# ---------------------------------------------------------------------------
# 2. Home-track advantage
# ---------------------------------------------------------------------------
def hta_lub(t: int) -> float:
    """Lublin learning-curve HTA for t-th home match (t=1..7)."""
    return 0.06 + 0.12 * (1.0 - np.exp(-0.35 * (t - 1)))


def build_fixtures() -> list[tuple[int, int, int]]:
    """
    Return list of (home_idx, away_idx, lub_home_match_no).
    Round-robin double: each pair home & away.
    lub_home_match_no is 1..7 when home is LUB else 0.
    """
    n = len(TEAM_ORDER)
    lub = TEAM_ORDER.index("LUB")
    pairs: list[tuple[int, int]] = []
    for i in range(n):
        for j in range(n):
            if i != j:
                pairs.append((i, j))

    # Order into 14 rounds of 4 matches (circle method-ish: just sequential ok for MC)
    # Assign LUB home match counters
    lub_home_count = 0
    fixtures: list[tuple[int, int, int]] = []
    for h, a in pairs:
        lub_t = 0
        if h == lub:
            lub_home_count += 1
            lub_t = lub_home_count
        fixtures.append((h, a, lub_t))
    assert lub_home_count == 7
    assert len(fixtures) == 56
    return fixtures


# ---------------------------------------------------------------------------
# 3. Match simulator (vectorized over seasons)
# ---------------------------------------------------------------------------
def _rider_indices_for_heat(heat_nums: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map 4 start numbers → (team_side 0=home/1=away, slot 0..6)."""
    sides = np.empty(4, dtype=np.int16)
    slots = np.empty(4, dtype=np.int16)
    for k, num in enumerate(heat_nums):
        num = int(num)
        if num >= 9:
            sides[k] = 0
            slots[k] = NUM_TO_SLOT[num]
        else:
            sides[k] = 1
            slots[k] = NUM_TO_SLOT[num]
    return sides, slots


# Precompute fixed-heat side/slot maps
_FIXED_SIDES = np.stack([_rider_indices_for_heat(h)[0] for h in HEATS_1_13])  # (13,4)
_FIXED_SLOTS = np.stack([_rider_indices_for_heat(h)[1] for h in HEATS_1_13])  # (13,4)


def simulate_matches_batch(
    theta_home: np.ndarray,
    theta_away: np.ndarray,
    track_noise: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate many matches in parallel.

    theta_home, theta_away : (N, 7)
    track_noise            : (N,) or scalar
    Returns home_pts, away_pts : (N,) race-point totals (0..45-ish)
    """
    n = theta_home.shape[0]
    noise = np.asarray(track_noise, dtype=np.float64)
    if noise.ndim == 0:
        noise = np.full(n, float(noise))

    home_pts = np.zeros(n, dtype=np.float64)
    away_pts = np.zeros(n, dtype=np.float64)
    # per-rider points for nominated heats
    rider_home = np.zeros((n, 7), dtype=np.float64)
    rider_away = np.zeros((n, 7), dtype=np.float64)

    # ---- heats 1–13 (fixed table) ----
    for h in range(13):
        sides = _FIXED_SIDES[h]
        slots = _FIXED_SLOTS[h]
        thetas4 = np.empty((n, 4), dtype=np.float64)
        for k in range(4):
            if sides[k] == 0:
                thetas4[:, k] = theta_home[:, slots[k]]
            else:
                thetas4[:, k] = theta_away[:, slots[k]]

        gumbel = rng.gumbel(0.0, 1.0, size=(n, 4)) * noise[:, None]
        scores = thetas4 + gumbel
        # points = number of rivals beaten = 3,2,1,0
        order = np.argsort(-scores, axis=1)  # best first
        pts = np.empty((n, 4), dtype=np.float64)
        # rank position -> points
        pts[np.arange(n)[:, None], order] = np.array([3.0, 2.0, 1.0, 0.0])

        for k in range(4):
            if sides[k] == 0:
                home_pts += pts[:, k]
                rider_home[:, slots[k]] += pts[:, k]
            else:
                away_pts += pts[:, k]
                rider_away[:, slots[k]] += pts[:, k]

    # ---- heats 14–15 (nominated) ----
    # heat 15: top-2 scorers each side; heat 14: next two
    order_h = np.argsort(-rider_home, axis=1)  # (n,7)
    order_a = np.argsort(-rider_away, axis=1)

    for heat_rank in (1, 0):  # first heat 14 (ranks 2,3), then heat 15 (ranks 0,1)
        # heat_rank=1 → slots at positions 2,3 ; heat_rank=0 → 0,1
        if heat_rank == 1:
            hi = order_h[:, 2:4]
            ai = order_a[:, 2:4]
        else:
            hi = order_h[:, 0:2]
            ai = order_a[:, 0:2]

        thetas4 = np.stack(
            [
                theta_home[np.arange(n), hi[:, 0]],
                theta_away[np.arange(n), ai[:, 0]],
                theta_home[np.arange(n), hi[:, 1]],
                theta_away[np.arange(n), ai[:, 1]],
            ],
            axis=1,
        )
        gumbel = rng.gumbel(0.0, 1.0, size=(n, 4)) * noise[:, None]
        scores = thetas4 + gumbel
        order = np.argsort(-scores, axis=1)
        pts = np.empty((n, 4), dtype=np.float64)
        pts[np.arange(n)[:, None], order] = np.array([3.0, 2.0, 1.0, 0.0])

        home_pts += pts[:, 0] + pts[:, 2]
        away_pts += pts[:, 1] + pts[:, 3]
        # (no need to update rider tallies further)

    return home_pts, away_pts


# ---------------------------------------------------------------------------
# 4. Season simulation
# ---------------------------------------------------------------------------
def simulate_seasons(
    theta_base: np.ndarray,
    fixtures: list[tuple[int, int, int]],
    n_seasons: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Run n_seasons full seasons.

    Returns arrays shaped (n_seasons, 8):
      rs_points, final_place, gold, medal, playoff, barez, relegated
    """
    n_teams = 8
    n_matches = len(fixtures)
    byd = TEAM_ORDER.index("BYD")

    # Precompute per-match home/away theta with HTA
    # theta_home_match[m, slot] — same across seasons (deterministic ratings)
    th_home = np.zeros((n_matches, 7), dtype=np.float64)
    th_away = np.zeros((n_matches, 7), dtype=np.float64)
    noise = np.zeros(n_matches, dtype=np.float64)
    home_idx = np.empty(n_matches, dtype=np.int16)
    away_idx = np.empty(n_matches, dtype=np.int16)

    for m, (h, a, lub_t) in enumerate(fixtures):
        home_idx[m] = h
        away_idx[m] = a
        th_home[m] = theta_base[h]
        th_away[m] = theta_base[a]
        if h == TEAM_ORDER.index("LUB"):
            th_home[m] = th_home[m] + hta_lub(lub_t)
        else:
            th_home[m] = th_home[m] + HTA_DEFAULT
        noise[m] = TRACK_NOISE_BYD if h == byd else TRACK_NOISE_DEFAULT

    # Expand across seasons: (n_seasons * n_matches, 7)
    N = n_seasons * n_matches
    th_h = np.broadcast_to(th_home[None, :, :], (n_seasons, n_matches, 7)).reshape(N, 7).copy()
    th_a = np.broadcast_to(th_away[None, :, :], (n_seasons, n_matches, 7)).reshape(N, 7).copy()
    nz = np.broadcast_to(noise[None, :], (n_seasons, n_matches)).reshape(N)

    print(f"Symulacja {n_seasons} symulacji × {n_matches} meczów = {N:,} meczów…")
    hp, ap = simulate_matches_batch(th_h, th_a, nz, rng)
    hp = hp.reshape(n_seasons, n_matches)
    ap = ap.reshape(n_seasons, n_matches)

    # ---- Regular season league points + bonus ----
    # 2 for win, 1 for draw; +1 bonus for better aggregate in home/away pair
    rs_pts = np.zeros((n_seasons, n_teams), dtype=np.float64)
    race_for = np.zeros((n_seasons, n_teams), dtype=np.float64)
    race_against = np.zeros((n_seasons, n_teams), dtype=np.float64)

    for m in range(n_matches):
        h, a = int(home_idx[m]), int(away_idx[m])
        h_score = hp[:, m]
        a_score = ap[:, m]
        # league points
        hw = h_score > a_score
        aw = a_score > h_score
        dr = h_score == a_score
        rs_pts[:, h] += 2.0 * hw + 1.0 * dr
        rs_pts[:, a] += 2.0 * aw + 1.0 * dr
        race_for[:, h] += h_score
        race_for[:, a] += a_score
        race_against[:, h] += a_score
        race_against[:, a] += h_score

    # Bonus: for each unordered pair, compare aggregates
    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            # find the two fixtures i vs j
            m_ij = None
            m_ji = None
            for m in range(n_matches):
                if home_idx[m] == i and away_idx[m] == j:
                    m_ij = m
                elif home_idx[m] == j and away_idx[m] == i:
                    m_ji = m
            assert m_ij is not None and m_ji is not None
            agg_i = hp[:, m_ij] + ap[:, m_ji]
            agg_j = ap[:, m_ij] + hp[:, m_ji]
            rs_pts[:, i] += (agg_i > agg_j).astype(np.float64)
            rs_pts[:, j] += (agg_j > agg_i).astype(np.float64)
            # draw on aggregate → no bonus (or both get 0)

    # Tie-break for ranking: rs_pts, then race_diff, then race_for
    race_diff = race_for - race_against
    # argsort: higher pts better → sort key descending
    # build composite key
    # rank_key higher = better
    rank_key = rs_pts * 1e9 + race_diff * 1e3 + race_for
    # places after RS: 0=best .. 7=worst
    rs_order = np.argsort(-rank_key, axis=1)  # (S, 8) team indices best→worst

    # ---- Play-offs / play-downs ----
    final_place = np.empty((n_seasons, n_teams), dtype=np.int16)

    def two_leg(team_a: np.ndarray, team_b: np.ndarray, seed_offset: int):
        """
        Vectorized two-leg tie. team_a has better RS seed (hosts 2nd leg).
        Returns winner index array (S,), loser index array (S,).
        Leg1: B home vs A away; Leg2: A home vs B away.
        """
        s = n_seasons
        # Leg 1: B home
        th_h1 = theta_base[team_b] + HTA_DEFAULT
        th_a1 = theta_base[team_a]
        # BYD noise if B is BYD
        nz1 = np.where(team_b == byd, TRACK_NOISE_BYD, TRACK_NOISE_DEFAULT).astype(np.float64)
        # LUB special HTA if B is LUB — use mid-curve value (~match 4)
        lub = TEAM_ORDER.index("LUB")
        th_h1 = th_h1.copy()
        th_h1[team_b == lub] = theta_base[lub] + hta_lub(4)

        rng1 = np.random.default_rng(RNG_SEED + seed_offset)
        # use sub-rng from main by jumping — simpler: pass sliced from parent via spawn
        h1, a1 = simulate_matches_batch(th_h1, th_a1, nz1, rng1)

        # Leg 2: A home
        th_h2 = theta_base[team_a] + HTA_DEFAULT
        th_a2 = theta_base[team_b]
        nz2 = np.where(team_a == byd, TRACK_NOISE_BYD, TRACK_NOISE_DEFAULT).astype(np.float64)
        th_h2 = th_h2.copy()
        th_h2[team_a == lub] = theta_base[lub] + hta_lub(5)
        rng2 = np.random.default_rng(RNG_SEED + seed_offset + 17)
        h2, a2 = simulate_matches_batch(th_h2, th_a2, nz2, rng2)

        agg_a = a1 + h2
        agg_b = h1 + a2
        # tie-break: better seed (A) wins if aggregate level
        a_wins = (agg_a > agg_b) | ((agg_a == agg_b))
        winners = np.where(a_wins, team_a, team_b)
        losers = np.where(a_wins, team_b, team_a)
        return winners, losers

    # Seeds
    s1 = rs_order[:, 0]
    s2 = rs_order[:, 1]
    s3 = rs_order[:, 2]
    s4 = rs_order[:, 3]
    s5 = rs_order[:, 4]
    s6 = rs_order[:, 5]
    s7 = rs_order[:, 6]
    s8 = rs_order[:, 7]

    # Semi: 1v4, 2v3
    w14, l14 = two_leg(s1, s4, 100)
    w23, l23 = two_leg(s2, s3, 200)
    # Final & 3rd place — better RS seed hosts second leg
    # Determine which finalist has better RS seed
    # rank position of each team
    rs_rank = np.empty((n_seasons, n_teams), dtype=np.int16)
    rs_rank[np.arange(n_seasons)[:, None], rs_order] = np.arange(8)

    def better_seed(ta, tb):
        # lower rs_rank is better
        a_better = rs_rank[np.arange(n_seasons), ta] < rs_rank[np.arange(n_seasons), tb]
        return np.where(a_better, ta, tb), np.where(a_better, tb, ta)

    f_a, f_b = better_seed(w14, w23)
    gold, silver = two_leg(f_a, f_b, 300)
    t_a, t_b = better_seed(l14, l23)
    bronze, fourth = two_leg(t_a, t_b, 400)

    # Play-down: 5v8, 6v7
    w58, l58 = two_leg(s5, s8, 500)
    w67, l67 = two_leg(s6, s7, 600)
    # winners stay (places 5/6 by RS), losers play for 7th (barez) vs 8th (relegation)
    d_a, d_b = better_seed(l58, l67)
    seventh, eighth = two_leg(d_a, d_b, 700)

    # Assign final places 5 & 6 to play-down winners by RS seed
    # better RS among w58/w67 → 5th, other → 6th
    p5, p6 = better_seed(w58, w67)

    final_place[np.arange(n_seasons), gold] = 1
    final_place[np.arange(n_seasons), silver] = 2
    final_place[np.arange(n_seasons), bronze] = 3
    final_place[np.arange(n_seasons), fourth] = 4
    final_place[np.arange(n_seasons), p5] = 5
    final_place[np.arange(n_seasons), p6] = 6
    final_place[np.arange(n_seasons), seventh] = 7
    final_place[np.arange(n_seasons), eighth] = 8

    return {
        "rs_points": rs_pts,
        "final_place": final_place,
        "gold": (final_place == 1).astype(np.float64),
        "medal": (final_place <= 3).astype(np.float64),
        "playoff": (final_place <= 4).astype(np.float64),
        "barez": (final_place == 7).astype(np.float64),
        "relegated": (final_place == 8).astype(np.float64),
    }


# ---------------------------------------------------------------------------
# 5. Results table + heatmap
# ---------------------------------------------------------------------------
def export_results(
    team_names: dict[str, str],
    results: dict[str, np.ndarray],
) -> pd.DataFrame:
    rows = []
    for ti, code in enumerate(TEAM_ORDER):
        rows.append(
            {
                "Kod": code,
                "Drużyna": team_names[code],
                "Śr. pkt runda zasadnicza": round(float(results["rs_points"][:, ti].mean()), 2),
                "% Play-off (Top 4)": round(100 * float(results["playoff"][:, ti].mean()), 1),
                "% Złoto": round(100 * float(results["gold"][:, ti].mean()), 1),
                "% Medal (Top 3)": round(100 * float(results["medal"][:, ti].mean()), 1),
                "% Baraż (7.)": round(100 * float(results["barez"][:, ti].mean()), 1),
                "% Spadek (8.)": round(100 * float(results["relegated"][:, ti].mean()), 1),
            }
        )
    out = pd.DataFrame(rows).sort_values(
        "Śr. pkt runda zasadnicza", ascending=False
    ).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    return out


def plot_heatmap(team_names: dict[str, str], final_place: np.ndarray) -> None:
    """Heatmap: P(place=1..8) for each team."""
    mat = np.zeros((8, 8), dtype=np.float64)
    for ti in range(8):
        for place in range(1, 9):
            mat[ti, place - 1] = (final_place[:, ti] == place).mean() * 100.0

    # order teams by P(gold) then playoff
    order = np.argsort(-mat[:, 0])
    mat = mat[order]
    labels = [f"{TEAM_ORDER[i]}\n{team_names[TEAM_ORDER[i]].split()[-1]}" for i in order]

    plt.figure(figsize=(10, 6.5))
    sns.heatmap(
        mat,
        annot=True,
        fmt=".1f",
        cmap="YlOrRd",
        xticklabels=[str(p) for p in range(1, 9)],
        yticklabels=labels,
        vmin=0,
        vmax=max(40, mat.max()),
        cbar_kws={"label": "% szans"},
    )
    plt.xlabel("Miejsce końcowe")
    plt.ylabel("Drużyna")
    plt.title("PGE Ekstraliga 2027 - prognoza Monte Carlo (10 000 symulacji)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=160)
    plt.close()
    print(f"Zapisano wykres: {OUT_PNG}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    print("=" * 64)
    print("PGE Ekstraliga 2027 - symulacja Monte Carlo")
    print("=" * 64)

    team_names, theta, _ = load_lineups(XLSX)
    fixtures = build_fixtures()
    rng = np.random.default_rng(RNG_SEED)

    import time

    t0 = time.perf_counter()
    results = simulate_seasons(theta, fixtures, N_SEASONS, rng)
    dt = time.perf_counter() - t0
    print(f"\nCzas symulacji: {dt:.2f} s")

    table = export_results(team_names, results)
    print("\n=== Tabela zbiorcza ===")
    print(table.to_string(index=False))
    print(f"\nZapisano CSV: {OUT_CSV}")

    plot_heatmap(team_names, results["final_place"])
    print("Gotowe.")


if __name__ == "__main__":
    main()
