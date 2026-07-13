# %% [markdown]
# # Joint genetic-interaction (GI) probability model — toy demo
#
# This is a runnable, scripted twin of a Jupyter notebook (cells are delimited
# with `# %%`, so it opens directly in VS Code / Jupytext / `jupytext --to
# notebook joint_model_demo.py`).
#
# It walks through the **joint mixture model** on a tiny, fully-shipped toy set:
# the 105 unordered gene pairs (+ 15 self-pairs) among 15 published genes,
# using **real** per-screen GI scores + standard errors from the two CRISPRi
# titration screens (100 ng/ml and 500 ng/ml ATc).
#
# What the joint model does, in one sentence: for each gene pair it combines the
# two per-screen GI scores (`delta_prime_median`) and their measurement error
# (`sd_delta_prime_median`) into a 5-component mixture — a correlated bivariate
# Normal *null* plus four quadrant "interaction" components — and returns the
# posterior probability that the pair is **aggravating** (both scores negative),
# **alleviating** (both positive), **discordant** (opposite signs), or
# **no interaction** (null).
#
# The demo runs in one of two modes:
#   * **Live** (a working CmdStan toolchain is installed): compile + sample the
#     primary Stan model on the 55 toy pairs, then extract the class
#     probabilities from the posterior.
#   * **Fallback** (no CmdStan): load the pre-derived golden checkpoint
#     `example_data/expected/merged_quad_me_trunc_halfsmeared_toy.tsv` so a reader
#     with no Stan toolchain can still step through every downstream cell.
#
# NOTE on interpretation: the golden checkpoint reflects the model **fit on the
# full published cohort** (thousands of pairs), which is what the paper reports.
# A *live* fit on only these 55 toy pairs is illustrative — the global
# parameters (mu, sigma, rho, mixture weights) are learned from far fewer points,
# so per-pair probabilities can differ from the checkpoint. That is expected and
# is exactly why the pipeline ships both a runnable chain and golden outputs.

# %%
# ── Cell 1: imports, paths, config ────────────────────────────────────────────
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Repo layout: this file lives in <repo>/notebooks/. Resolve the repo root so we
# can import the slimmed joint runner and locate the Stan file + toy data.
try:
    HERE = Path(__file__).resolve().parent
except NameError:            # running inside a notebook kernel (no __file__)
    HERE = Path.cwd()
REPO_ROOT = HERE.parent if HERE.name == "notebooks" else HERE
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Toy data + model paths.
GI_INPUT_DIR = REPO_ROOT / "example_data" / "gi_input"
EXPECTED_DIR = REPO_ROOT / "example_data" / "expected"
EXP1_TSV = GI_INPUT_DIR / "result_summary_long_df_exp1_toy.tsv"   # 100 ng/ml ATc
EXP2_TSV = GI_INPUT_DIR / "result_summary_long_df_exp2_toy.tsv"   # 500 ng/ml ATc
GOLDEN_MERGED = EXPECTED_DIR / "merged_quad_me_trunc_halfsmeared_toy.tsv"
STAN_FILE = REPO_ROOT / "joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan"

# Where demo figures are written (also shown inline in a notebook).
OUT_DIR = HERE / "demo_output"
OUT_DIR.mkdir(exist_ok=True)

# ── Pinned defaults (match the production pipeline exactly) ───────────────────
WINSORIZE_PCT = 0.10          # clip support to [0.10, 99.90] percentile per axis
N_CHAINS = 8
N_WARMUP = 1000
N_SAMPLING = 1000
SEED = 456
CALL_THRESHOLD = 0.50         # dominant-class call threshold
STRICT_THRESHOLD = 0.95       # stricter "confident" cutoff (mentioned below)

# ── Class colors (identical to the paper's revision figures) ──────────────────
CLASS_COLORS = {
    "aggravating": "#2e7d32",   # green  — both screens negative (buffering/NEG-GI)
    "alleviating": "#7b1fa2",   # purple — both screens positive (POS-GI)
    "discordant":  "#ff7f0e",   # orange — opposite signs across screens
    "no_interaction": "#cccccc",  # grey  — null
}
# prob column -> class key, in the priority order used to assign a dominant call.
CLASS_PROB_COLS = [
    ("aggravating", "prob_aggravating"),
    ("alleviating", "prob_alleviating"),
    ("discordant",  "prob_discordant"),
]
X_LABEL = "GI score — 100 ng/ml ATc"
Y_LABEL = "GI score — 500 ng/ml ATc"

# Matplotlib is optional; the demo still prints all tables without it.
try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False
    print("matplotlib not available — scatter cells will be skipped, tables still print.")

print(f"Repo root : {REPO_ROOT}")
print(f"Stan file : {STAN_FILE}  (exists: {STAN_FILE.exists()})")


# %%
# ── Cell 2: detect CmdStan; decide live-fit vs golden-checkpoint fallback ─────
# We only run the live Stan chain if a compiled CmdStan toolchain is actually
# installed. cmdstanpy imports fine without it, so we probe cmdstan_path().
HAS_CMDSTAN = False
try:
    import cmdstanpy
    cmdstanpy.cmdstan_path()      # raises if CmdStan is not installed/configured
    HAS_CMDSTAN = STAN_FILE.exists()
except Exception as exc:
    print(f"CmdStan not usable ({type(exc).__name__}: {exc}).")

print(f"HAS_CMDSTAN = {HAS_CMDSTAN}  ->  "
      + ("LIVE fit on the 55 toy pairs" if HAS_CMDSTAN
         else "FALLBACK to golden checkpoint"))


# %%
# ── Cell 3: reusable helpers ──────────────────────────────────────────────────
# Prefer the production `build_stan_data` from the slimmed joint runner so the
# toy Stan JSON is constructed byte-for-byte the way the paper's pipeline builds
# it (same winsorization, same fixed origin, same hyperpriors). Fall back to a
# faithful local copy if that module is not importable in this checkout.
build_stan_data = None
for _mod in ("run_joint_model", "run_joint_mixture_models"):
    try:
        build_stan_data = __import__(_mod).build_stan_data
        print(f"Using build_stan_data() imported from {_mod}.")
        break
    except Exception:
        continue

if build_stan_data is None:
    print("Joint runner not importable — using local build_stan_data() copy.")

    def build_stan_data(merged, include_se=False, winsorize_pct=None):
        """Build the Stan data dict for the joint quadrant models.

        Faithful copy of run_joint_mixture_models.build_stan_data: fixed origin
        at 0, hyperpriors mu_mu=0/sigma_mu=2/a_sigma=2/b_sigma=0.1, and optional
        winsorization of the uniform support to [pct, 100-pct] per axis.
        """
        y1 = merged["delta_prime_median_exp1"].values.copy()
        y2 = merged["delta_prime_median_exp2"].values.copy()

        if winsorize_pct is not None:
            lo1, hi1 = np.percentile(y1, [winsorize_pct, 100 - winsorize_pct])
            lo2, hi2 = np.percentile(y2, [winsorize_pct, 100 - winsorize_pct])
            n_clip1 = int(((y1 < lo1) | (y1 > hi1)).sum())
            n_clip2 = int(((y2 < lo2) | (y2 > hi2)).sum())
            y1 = np.clip(y1, lo1, hi1)
            y2 = np.clip(y2, lo2, hi2)
            print(f"  Winsorized to [{winsorize_pct}, {100 - winsorize_pct}] pct:")
            print(f"    y1: [{lo1:.2f}, {hi1:.2f}], clipped {n_clip1} pairs")
            print(f"    y2: [{lo2:.2f}, {hi2:.2f}], clipped {n_clip2} pairs")

        data = {
            "N": len(merged),
            "y1": y1.tolist(),
            "y2": y2.tolist(),
            "min_y1": float(np.min(y1)),
            "max_y1": float(np.max(y1)),
            "min_y2": float(np.min(y2)),
            "max_y2": float(np.max(y2)),
            "origin_y1": 0.0,
            "origin_y2": 0.0,
            "mu_mu": 0,
            "sigma_mu": 2,
            "a_sigma": 2,
            "b_sigma": 0.1,
            "rho_theta": 0.5,    # present in the data block but UNUSED by the model
            "kappa_theta": 2,    # present in the data block but UNUSED by the model
        }
        if include_se:
            data["se1"] = merged["sd_delta_prime_median_exp1"].values.tolist()
            data["se2"] = merged["sd_delta_prime_median_exp2"].values.tolist()
        return data


def _pick(m, base):
    """Return column `base_exp1` if present, else the unsuffixed `base`."""
    if f"{base}_exp1" in m.columns:
        return m[f"{base}_exp1"]
    return m[base]


def load_toy_inputs(path1, path2):
    """Load the two per-screen toy TSVs and inner-join on orf_pair.

    Returns a tidy frame with one row per shared pair and unified columns:
        orf1, orf2, name1, name2, orf_pair,
        gi_score_exp1, gi_score_exp2, se_exp1, se_exp2,
        and the suffixed delta_prime_median_* / sd_delta_prime_median_* columns
        that build_stan_data() consumes directly.
    """
    keep = ["orf1", "orf2", "orf_pair", "name1", "name2",
            "delta_prime_median", "sd_delta_prime_median",
            "prob_interaction_median", "correlation"]

    def _read(p):
        d = pd.read_csv(p, sep="\t")
        d = d.loc[:, ~d.columns.str.startswith("Unnamed")]  # drop the index col
        if "orf_pair" not in d.columns:                     # reconstruct if missing
            d["orf_pair"] = ["_".join(sorted([a, b]))
                             for a, b in zip(d["orf1"], d["orf2"])]
        return d[[c for c in keep if c in d.columns]]

    d1, d2 = _read(path1), _read(path2)
    m = d1.merge(d2, on="orf_pair", suffixes=("_exp1", "_exp2"))

    tidy = pd.DataFrame({
        "orf1": _pick(m, "orf1"),
        "orf2": _pick(m, "orf2"),
        "name1": _pick(m, "name1"),
        "name2": _pick(m, "name2"),
        "orf_pair": m["orf_pair"],
        "gi_score_exp1": m["delta_prime_median_exp1"],
        "gi_score_exp2": m["delta_prime_median_exp2"],
        "se_exp1": m["sd_delta_prime_median_exp1"],
        "se_exp2": m["sd_delta_prime_median_exp2"],
    })
    # Keep the raw suffixed columns too, so build_stan_data can be called on `m`.
    return tidy, m


def assign_call(df, threshold=CALL_THRESHOLD):
    """Add a 'call' column: first class whose prob >= threshold, else null.

    Mirrors the paper's classification scatter (np.select over the interaction
    classes, defaulting to the null class).
    """
    conds = [df[col].values >= threshold for _, col in CLASS_PROB_COLS]
    choices = [cls for cls, _ in CLASS_PROB_COLS]
    df = df.copy()
    df["call"] = np.select(conds, choices, default="no_interaction")
    return df


# %%
# ── Cell 4: load the toy inputs and inner-join on orf_pair ────────────────────
tidy, merged_raw = load_toy_inputs(EXP1_TSV, EXP2_TSV)
print(f"Loaded {len(tidy)} shared toy pairs (expected 55).")
print(f"  gi_score_exp1 range: [{tidy['gi_score_exp1'].min():.2f}, "
      f"{tidy['gi_score_exp1'].max():.2f}]")
print(f"  gi_score_exp2 range: [{tidy['gi_score_exp2'].min():.2f}, "
      f"{tidy['gi_score_exp2'].max():.2f}]")
print(f"  se_exp1 range:       [{tidy['se_exp1'].min():.3f}, "
      f"{tidy['se_exp1'].max():.3f}]")
print(f"  se_exp2 range:       [{tidy['se_exp2'].min():.3f}, "
      f"{tidy['se_exp2'].max():.3f}]")
tidy.head(10)


# %%
# ── Cell 5: raw 2D scatter of the two per-screen GI scores ────────────────────
# Before any modeling: just plot screen-1 vs screen-2 GI scores. The joint model
# is essentially a probabilistic reading of *where in this plane* each pair sits.
if HAS_MPL:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.axvline(0, color="black", lw=0.6, alpha=0.4)
    ax.scatter(tidy["gi_score_exp1"], tidy["gi_score_exp2"],
               s=45, c="#377eb8", edgecolor="white", linewidth=0.5, zorder=3)
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title("Raw per-screen GI scores (55 toy pairs)")
    lim = max(abs(np.r_[tidy["gi_score_exp1"], tidy["gi_score_exp2"]]).max() * 1.1, 1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "raw_scatter.png", dpi=150)
    plt.show()
    print(f"Saved {OUT_DIR / 'raw_scatter.png'}")
else:
    print("Skipping raw scatter (matplotlib unavailable).")


# %%
# ── Cell 6: build the winsorized + SE Stan JSON ───────────────────────────────
# The measurement-error models take the per-pair SEs (se1, se2) and winsorized
# uniform support. This is the exact JSON the production runner feeds to Stan.
stan_data = build_stan_data(merged_raw, include_se=True, winsorize_pct=WINSORIZE_PCT)
json_path = OUT_DIR / "example_data_trunc_w010.json"
with open(json_path, "w") as fh:
    json.dump(stan_data, fh)

print(f"Wrote Stan data JSON -> {json_path}")
print(f"  N={stan_data['N']}")
print(f"  y1 support=[{stan_data['min_y1']:.2f}, {stan_data['max_y1']:.2f}]  "
      f"y2 support=[{stan_data['min_y2']:.2f}, {stan_data['max_y2']:.2f}]")
print(f"  origin=({stan_data['origin_y1']}, {stan_data['origin_y2']}); "
      f"hyperpriors: mu_mu={stan_data['mu_mu']}, sigma_mu={stan_data['sigma_mu']}, "
      f"a_sigma={stan_data['a_sigma']}, b_sigma={stan_data['b_sigma']}")


# %%
# ── Cell 7: fit live (if CmdStan present) OR load the golden checkpoint ────────
# The Stan generated-quantities block emits per-pair log-probabilities for each
# component; we exponentiate and average across posterior draws to get the class
# probabilities. Column mapping (Stan -> paper):
#     log_pZ1   -> prob_interaction   (any interaction)
#     log_p_aggr-> prob_aggravating   log_p_allev -> prob_alleviating
#     log_p_disc-> prob_discordant    log_p_null  -> prob_no_interaction
#     log_p_conc-> prob_concordant    (= aggravating + alleviating)
PROB_MAP = {
    "prob_interaction":    "log_pZ1",
    "prob_aggravating":    "log_p_aggr",
    "prob_alleviating":    "log_p_allev",
    "prob_discordant":     "log_p_disc",
    "prob_no_interaction": "log_p_null",
    "prob_concordant":     "log_p_conc",
}

if HAS_CMDSTAN:
    print("Compiling Stan model (first run only; cached afterwards)...")
    model = cmdstanpy.CmdStanModel(stan_file=str(STAN_FILE))

    # Same reasonable inits the production runner uses for the simplex model.
    y1 = np.asarray(stan_data["y1"])
    y2 = np.asarray(stan_data["y2"])
    inits = {
        "mu1": 0.0, "mu2": 0.0,
        "sigma1": float(np.std(y1)), "sigma2": float(np.std(y2)),
        "rho": 0.15,
        "mix_weights": [0.05, 0.05, 0.90],
    }

    print(f"Sampling ({N_CHAINS} chains x {N_WARMUP} warmup + {N_SAMPLING} "
          f"sampling, seed={SEED})...")
    fit = model.sample(
        data=str(json_path),
        chains=N_CHAINS,
        iter_warmup=N_WARMUP,
        iter_sampling=N_SAMPLING,
        seed=SEED,
        inits=inits,
        show_progress=False,
    )

    # posterior-mean class probabilities (matches how the paper's summaries reduce
    # the per-pair draws: mean of exp(log_p) across draws).
    # Row order of the posterior draws matches merged_raw (the frame used to
    # build the Stan JSON), so we can attach orf_pair directly.
    probs = {"orf_pair": merged_raw["orf_pair"].values}
    for out_col, stan_var in PROB_MAP.items():
        probs[out_col] = np.exp(fit.stan_variable(stan_var)).mean(axis=0)
    probs = pd.DataFrame(probs)

    results = tidy.merge(probs, on="orf_pair", how="inner")
    source_note = "LIVE Stan fit on the 55 toy pairs"

    # Quick posterior sanity read on the global parameters.
    for name in ["mu1", "mu2", "sigma1", "sigma2", "rho"]:
        v = fit.stan_variable(name)
        print(f"    {name:7s} median={np.median(v):7.3f}  "
              f"[{np.percentile(v, 2.5):.3f}, {np.percentile(v, 97.5):.3f}]")
else:
    print(f"Loading golden checkpoint: {GOLDEN_MERGED}")
    golden = pd.read_csv(GOLDEN_MERGED, sep="\t")
    golden["orf_pair"] = golden["orf1"].astype(str) + "_" + golden["orf2"].astype(str)
    prob_cols = list(PROB_MAP.keys())
    results = golden[["orf1", "orf2", "name1", "name2", "orf_pair",
                      "gi_score_exp1", "gi_score_exp2",
                      "se_exp1", "se_exp2"] + prob_cols].copy()
    source_note = "GOLDEN checkpoint (full-cohort fit)"

print(f"\nProbabilities source: {source_note}  ({len(results)} pairs)")
results = assign_call(results, threshold=CALL_THRESHOLD)


# %%
# ── Cell 8: classification scatter — GI scores colored by dominant class ──────
# Same encoding as the paper figure: aggravating green, alleviating purple,
# discordant orange, null grey. Non-null toy pairs are labeled by gene names.
if HAS_MPL:
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axhline(0, color="black", lw=0.6, alpha=0.4)
    ax.axvline(0, color="black", lw=0.6, alpha=0.4)

    counts = results["call"].value_counts()
    draw_order = ["no_interaction", "discordant", "aggravating", "alleviating"]
    for cls in draw_order:
        sub = results[results["call"] == cls]
        if len(sub) == 0:
            continue
        ax.scatter(sub["gi_score_exp1"], sub["gi_score_exp2"],
                   s=55, c=CLASS_COLORS[cls], edgecolor="white", linewidth=0.5,
                   label=f"{cls} ({counts.get(cls, 0)})",
                   zorder=3 if cls != "no_interaction" else 2)

    # Label the called interactions (small toy set -> readable).
    for _, r in results[results["call"] != "no_interaction"].iterrows():
        ax.annotate(f"{r['name1']}-{r['name2']}",
                    (r["gi_score_exp1"], r["gi_score_exp2"]),
                    textcoords="offset points", xytext=(4, 4),
                    fontsize=7, fontstyle="italic")

    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(Y_LABEL)
    ax.set_title(f"Joint classification (dominant class, P >= {CALL_THRESHOLD})")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    lim = max(abs(np.r_[results["gi_score_exp1"], results["gi_score_exp2"]]).max() * 1.1, 1)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "classification_scatter.png", dpi=150)
    plt.show()
    print(f"Saved {OUT_DIR / 'classification_scatter.png'}")
else:
    print("Skipping classification scatter (matplotlib unavailable).")


# %%
# ── Cell 9: per-pair 4-class probability table + call at T=0.5 ─────────────────
table = results.copy()
table["pair"] = table["name1"].astype(str) + "-" + table["name2"].astype(str)
show_cols = ["pair", "gi_score_exp1", "gi_score_exp2",
             "prob_aggravating", "prob_alleviating",
             "prob_discordant", "prob_no_interaction", "call"]
table = table[show_cols].sort_values("prob_no_interaction").reset_index(drop=True)

pd.set_option("display.max_rows", None)
pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: f"{v:.3f}")

print(f"Per-pair 4-class probabilities and call at T={CALL_THRESHOLD} "
      f"[{source_note}]\n")
print(table.to_string(index=False))

print("\nCall breakdown (T=0.50):")
print(results["call"].value_counts().to_string())

# Stricter, "confident" cutoff for reference (paper reports both 0.5 and 0.95).
strict = assign_call(results, threshold=STRICT_THRESHOLD)
print(f"\nCall breakdown at the stricter confident cutoff (T={STRICT_THRESHOLD}):")
print(strict["call"].value_counts().to_string())


# %%
# ── Cell 10: recap ────────────────────────────────────────────────────────────
# * The joint model turns a pair's two per-screen GI scores + SEs into posterior
#   probabilities over {aggravating, alleviating, discordant, no interaction}.
# * A pair is "called" as the first class whose probability clears the threshold
#   (0.5 by default; 0.95 for a confident set), else it stays null.
# * Live-fitting on 120 pairs is a teaching illustration; the shipped golden
#   checkpoint (full-cohort fit) is what the paper reports. To reproduce the
#   full analysis, run the joint runner on the complete per-screen tables rather
#   than the toy cut.
print("Demo complete.")
print(f"  probabilities from: {source_note}")
print(f"  figures written to: {OUT_DIR}")
