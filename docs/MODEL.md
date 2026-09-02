# The joint genetic-interaction (GI) probability model

This document is the method + equations reference for the joint model that combines
two *M. tuberculosis* CRISPRi genetic-interaction screens into a single, calibrated
per-pair probability of interaction. It is written to be read alongside the Stan
sources and the runnable toy pipeline in this repository. Every equation below is
transcribed directly from the Stan program that produces the paper's numbers; file
and line references are given so a reader can check the code against the math.

**Primary model file:**
[`joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan`](../joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan)
(internal tag `quad_me_trunc_halfsmeared`, signing `signed_maxmag`).

Contents:

1. [Two-screen design and why a joint model](#1-two-screen-design-and-why-a-joint-model)
2. [Per-guide-pair two-line fitness model and Y25](#2-per-guide-pair-two-line-fitness-model-and-y25)
3. [δ′: the GI score, GAM de-trending, aggregation, and its SE](#3-δ-the-gi-score-gam-de-trending-aggregation-and-its-se)
4. [Per-screen 1-D Normal/Uniform mixture → P(interaction)](#4-per-screen-1-d-normaluniform-mixture--pinteraction)
5. [The joint 5-component half-smeared quadrant mixture](#5-the-joint-5-component-half-smeared-quadrant-mixture)
6. [Signing, thresholding, and the call logic](#6-signing-thresholding-and-the-call-logic)
7. [Interpretation: probability is classification confidence, not effect size](#7-interpretation-probability-is-classification-confidence-not-effect-size)

Notation: a *guide pair* is a physical dual-sgRNA construct; an *ORF pair* (gene
pair) aggregates all guide pairs targeting the same two genes. `orf_pair` is the
canonical sorted string `"orf1_orf2"` (`orf1 < orf2`). "Exp1" is the 100 ng/ml ATc
screen; "Exp2" is the 500 ng/ml ATc screen. δ′ (delta-prime) is the GI score.

---

## 1. Two-screen design and why a joint model

The two screens are the **same** library of dual-guide constructs assayed at **two
anhydrotetracycline (ATc) doses**, which set the strength of dCas9 knockdown:

| Screen | ATc dose | Knockdown | Role |
|--------|----------|-----------|------|
| Exp1   | 100 ng/ml | milder    | lower on-target repression, generally quieter |
| Exp2   | 500 ng/ml | stronger  | deeper repression, larger effects and more noise per pair |

Because the ATc dose changes how hard each gene is knocked down, the two screens
have **different noise floors**: the standard error attached to a given ORF pair's
GI score differs between Exp1 and Exp2, and the range of resolvable effect sizes
differs too. A pair with a moderate true interaction can sit *above* one screen's
noise floor and *below* the other's.

Analysing the two screens independently and then intersecting hit lists throws away
this structure and produces the artifact a reviewer flagged: the GI **scores** are
well correlated between screens (Pearson R ≈ 0.82), but the independently computed
**probabilities** correlate much less (R² ≈ 0.44). That is not disagreement about
the biology — it is a *threshold shift*. The single-screen decision boundary (the
|δ′| at which P(interaction) crosses 0.5) sits at roughly |δ′| ≈ 2.6 in Exp1 versus
|δ′| ≈ 4.4 in Exp2 because of the different noise floors, so pairs whose |δ′| falls
between those two shoulders get a high probability in one screen and a low one in
the other.

The **joint model** addresses this directly. It ingests both screens' GI scores
*and their per-pair standard errors* for every ORF pair and fits **one** mixture in
the 2-D (δ′₍Exp1₎, δ′₍Exp2₎) plane. A pair is classified from the joint evidence, and
the model's own measurement-error term lets each screen contribute in proportion to
its precision. Because the null component is a *correlated* bivariate Normal, the
model can also learn that the two screens' noise is itself correlated, which is what
separates "consistently small in both screens" (null) from "consistently large in
both screens" (a real interaction).

---

## 2. Per-guide-pair two-line fitness model and Y25

Upstream of everything below is the per-screen fitness pipeline
([`gi_scoring.py`](../gi_scoring.py); trajectory helpers in
[`logfc_tools.py`](../logfc_tools.py)). For each guide pair it fits the depletion
trajectory — log₂ fold-change `y` in construct abundance versus number of bacterial
generations `x` — with a **two-line (hinge / broken-stick) mean** and a
heavy-tailed Student-t likelihood (`gi_scoring.py`, `_create_default_stan_model`;
the standalone model is
[`singlepair_splits_delta_twoline_model_v1.stan`](../singlepair_splits_delta_twoline_model_v1.stan)):

$$
\mu(x)=
\begin{cases}
\alpha_\ell + \beta_\ell\,x & x \le \gamma \quad\text{(lag line)}\\[4pt]
(\alpha_\ell + \beta_\ell\,\gamma) + \beta_e\,(x-\gamma) & x > \gamma \quad\text{(established line)}
\end{cases}
\qquad
y \sim \text{Student-}t(\nu_y,\ \mu(x),\ \sigma)
$$

- $\gamma$ is the changepoint (generation at which the trajectory switches slope),
- $\beta_\ell$ is the early/lag slope, $\beta_e$ the established (steady-state)
  depletion slope, $\alpha_\ell$ the intercept,
- Student-t (with dof $\nu_y$) makes the fit robust to outlier timepoints.

Priors (from the embedded Stan program): $\alpha_\ell\sim N(0,1)$,
$\beta_e\sim N(-0.2,0.5)$, $\beta_\ell\sim N(0,0.2)$, $\gamma\sim N(4,2)$,
$\sigma\sim N(0.5,1)$, $\nu_y\sim N(3,1)$.

The single fitness readout extracted from the fit is **Y25**, the model-predicted
log₂FC at **generation 25** (a fixed late timepoint on the established line):

$$
\text{Y25} = (\alpha_\ell + \beta_\ell\,\gamma) + \beta_e\,(25-\gamma).
$$

Y25 is computed as a `generated quantity`, so it carries a full posterior (mean, sd,
2.5 %/97.5 % quantiles) per guide pair. A strongly depleted construct has a large
negative Y25; a neutral construct has Y25 near 0.

---

## 3. δ′: the GI score, GAM de-trending, aggregation, and its SE

**Raw GI score (per guide pair).** A genetic interaction is a departure from the
*additive* (log-additive in fitness) expectation. For a double-mutant construct
targeting genes A and B, the two "single-mutant" references are the same guides paired
with a non-targeting **Negative** guide (construct-id grammar `A:nameA_SEQ_B:nameB_SEQ`,
with the neutral side written literally as `Negative_SEQ`). The interaction score is

$$
\delta'_{\text{gp}} \;=\; \text{Y25}_{AB} \;-\; \bigl(\text{Y25}_{A\text{–Neg}} + \text{Y25}_{B\text{–Neg}}\bigr),
$$

i.e. observed double-mutant fitness minus the sum of the two single-mutant fitnesses
(`gi_scoring.py`, `calculate_y25_delta_scores`; the sum of singles is stored as
`y25_expected`). $\delta' < 0$ ⇒ the double mutant is *more* depleted than expected
(**aggravating / synthetic-sick**); $\delta' > 0$ ⇒ *less* depleted than expected
(**alleviating / buffering**).

**GAM de-trending against expected fitness.** The raw δ′ has a systematic dependence
on how depleted the pair already is: constructs near the assay's dynamic-range floor
show a fitness-dependent bias. We remove it by regressing δ′ on the expected fitness
with a smoothing spline and keeping the **residual** as the corrected score
([`gam_correction_standalone.R`](../gam_correction_standalone.R), mgcv; a `pygam`
fallback lives in `gi_scoring.py::apply_gam_correction_python`):

$$
\delta'_{\text{gp}} \sim s(\text{expected}), \qquad
\delta'_{\text{corrected}} = \delta'_{\text{gp}} - \widehat{s}(\text{expected}),
$$

with `expected` $=\text{Y25}_{A\text{–Neg}}+\text{Y25}_{B\text{–Neg}}$, a thin-plate
spline basis (`k = 20`), and smoothing chosen by ML. This centers the null cloud at 0
across the whole fitness range so the mixture's origin is meaningful.

**Guide-pair → ORF-pair aggregation.** Each ORF pair is measured by several guide
pairs (the guide-vs-guide combinations for the two genes). Pooling the corrected
δ′ posteriors across those guide pairs gives one estimate per ORF pair per screen:

- `delta_prime_median` = the **median** of the pooled δ′ posterior — this is the
  reported **GI score** for the ORF pair (`gi_score_exp1`, `gi_score_exp2` in the
  merged table);
- `sd_delta_prime_median` = the **standard deviation** of that same pooled posterior
  — the **standard error** of the GI score.

**SE identity used downstream.** The joint model's measurement-error inputs are
*exactly* these SEs — there is no separate noise estimate:

$$
\texttt{se1} \equiv \texttt{sd\_delta\_prime\_median}_{\text{Exp1}}, \qquad
\texttt{se2} \equiv \texttt{sd\_delta\_prime\_median}_{\text{Exp2}}.
$$

(The same column also defines the convenience flag `gi_score_overlaps_zero` via
$|\delta'| < 1.96\cdot\texttt{sd\_delta\_prime\_median}$ in
[`merge_joint_results.py`](../merge_joint_results.py).) These SEs
are what let the joint model weight each screen by its precision, per ORF pair.

> **Critical data note.** Only the HPC copies of `result_summary_long_df_*.tsv`
> carry `sd_delta_prime_median`; the `paper/data/raw_outputs` copies do **not**. The
> toy inputs in `example_data/gi_input/` are cut from the HPC copies so the SE column is
> present.

---

## 4. Per-screen 1-D Normal/Uniform mixture → P(interaction)

Before the joint fit, each screen is classified on its own with a **1-D Normal /
Uniform mixture** ([`normal_uniform_mix.stan`](../normal_uniform_mix.stan);
driver [`run_per_screen_mixture.py`](../run_per_screen_mixture.py)) — the model
the individual screens were called with. This produces the per-screen
`prob_interaction_median` columns (`prob_interaction_median_exp1/exp2`) that the
joint model reports alongside its own probabilities.

Each ORF pair's score $y=\delta'$ is either **null** (a Normal centered near 0,
$N(\mu,\sigma)$) or **interacting** (a broad Uniform over the winsorized support
$[\text{min}_y,\text{max}_y]$). With mixing weight $\theta$ (the interacting
fraction), the per-pair log-density is
$\log\!\bigl[(1-\theta)\,N(y\mid\mu,\sigma)+\theta\,\mathrm{Unif}(y)\bigr]$ and the
posterior interaction probability is

$$
P(\text{interaction}\mid y)=\frac{\theta\,\mathrm{Unif}(y)}
{(1-\theta)\,N(y\mid\mu,\sigma)+\theta\,\mathrm{Unif}(y)}.
$$

**Measurement-error variant — the joint model's building block.** The joint model
(§5) additionally folds in each pair's own standard error `se`. Its 1-D analogue,
[`univariate_normal_uniform_mix_me.stan`](../univariate_normal_uniform_mix_me.stan)
(select with `--stan`), replaces the two components above with ME-aware versions:

- **Null:** $N(\mu,\ s_{\text{tot}})$ with variance inflated by the measurement
  error, $s_{\text{tot}}=\sqrt{\sigma^2+se^2}$.
- **Interaction:** the Uniform **smeared** by the Gaussian ME kernel — the
  probability that a true effect drawn Uniform on $[a,b]$, observed with Gaussian
  noise `se`, lands at $y$:

$$
p_{\text{int}}(y)=\frac{\Phi\!\left(\frac{b-y}{se}\right)-\Phi\!\left(\frac{a-y}{se}\right)}{b-a}.
$$

Two features of this ME treatment recur in the joint model and drive the whole analysis:

- **ME inflation** ($s_{\text{tot}}=\sqrt{\sigma^2+se^2}$) means a noisy pair needs a
  *larger* |δ′| to be called — the decision boundary moves with the noise floor.
  This is precisely why the same true interaction can be a hit in one screen and not
  the other (§1).
- **Winsorizing** the support tightens the Uniform so that its plateau density is a
  fair competitor to the Normal; see the note in §5.

---

## 5. The joint 5-component half-smeared quadrant mixture

The primary model lifts the 1-D idea into the (δ′₍Exp1₎, δ′₍Exp2₎) plane and, crucially,
splits the interaction density by **quadrant** so the model reports *direction*, not
just presence. Data $y_1,y_2$ are the two screens' GI scores; `se1,se2` their SEs.
The mixture has **five components**: one correlated null plus one density per quadrant.

### 5.1 Components and quadrant → biology map

With the origin fixed at 0 on both axes (`origin_y1 = origin_y2 = 0`), sign of each
coordinate is the sign of that screen's GI score:

| Component | Region | Biology | Mixture weight |
|-----------|--------|---------|----------------|
| Null | around origin | no interaction | $\theta_{\text{null}}$ |
| Q4 | $y_1<0,\ y_2<0$ | **aggravating** (both synthetic-sick) | $\theta_{\text{conc}}$ |
| Q2 | $y_1>0,\ y_2>0$ | **alleviating** (both buffering) | $\theta_{\text{conc}}$ |
| Q1 | $y_1<0,\ y_2>0$ | **discordant** (opposite signs) | $\theta_{\text{disc}}$ |
| Q3 | $y_1>0,\ y_2<0$ | **discordant** (opposite signs) | $\theta_{\text{disc}}$ |

Aggravating (Q4) and alleviating (Q2) together are **concordant** interactions — the
two screens agree on sign. Q1/Q3 are **discordant** — the screens disagree on sign,
which for a real biological interaction should be rare and is treated as its own class.

### 5.2 The correlated, measurement-error-inflated null

The null is a bivariate Normal whose per-axis variance is inflated by each pair's own
measurement error and whose correlation is a **free shared parameter** `rho`
(model block, lines 144–154):

$$
s_{1,\text{tot}}=\sqrt{\sigma_1^2+se_1^2}, \qquad
s_{2,\text{tot}}=\sqrt{\sigma_2^2+se_2^2},
$$

$$
\rho_{\text{eff}}=\rho\,\frac{\sigma_1\,\sigma_2}{s_{1,\text{tot}}\,s_{2,\text{tot}}},
\qquad
z_k=\frac{y_k-\mu_k}{s_{k,\text{tot}}},
$$

$$
\log p_{\text{null}} = \log\theta_{\text{null}}
-\log(2\pi)-\log s_{1,\text{tot}}-\log s_{2,\text{tot}}
-\tfrac12\log(1-\rho_{\text{eff}}^2)
-\frac{z_1^2-2\rho_{\text{eff}}z_1z_2+z_2^2}{2(1-\rho_{\text{eff}}^2)}.
$$

Note the **attenuation**: because $s_{k,\text{tot}}\ge\sigma_k$, the *observed*
correlation $\rho_{\text{eff}}$ is always shrunk toward 0 relative to the latent
correlation $\rho$. Fitting `rho` freely therefore recovers the true correlation of
the null fitness noise between screens rather than the measurement-diluted one — this
is the mechanism that lets the joint model reconcile "well-correlated scores, less-
correlated single-screen probabilities" (§1, §7).

### 5.3 Quadrant interaction densities: "half-smeared"

Each quadrant density is a **product of two one-sided half-axis densities**, one per
screen. The design question is what shape to use on each half-axis. Three options were
tried; the header of the primary `.stan` file records why the third won:

- **Fully smeared (v1, `quad_me_trunc`)** convolves a Uniform on $[a,b]$ with the
  Gaussian ME kernel on **both** ends. This tapers the density near the winsorized
  outer cap, halving the plateau there, so pairs pinned to the edge of the support by
  winsorizing can *fail* the 0.95 hit threshold despite being confidently in-quadrant.
- **No smearing (`quad_null_me_trunc`)** fixes the outer taper but makes a **hard
  edge at the origin**: a pair like $(-10, 0.1)$ is called 100 % discordant even
  though ME says the $+0.1$ could easily be $-0.1$.
- **Half-smeared (this model)** keeps the origin soft (a graceful quadrant boundary
  through the noise) but removes the outer taper (a flat plateau out to the cap).

For the right half-axis of $y_1$ (support $[0, \text{max}_{y_1}]$, width $b=\text{max}_{y_1}-0$),
the half-smeared density is (transformed-data block, lines 91–106):

$$
f(y_1)=\frac{\Phi\!\left(\frac{y_1-0}{se_1}\right)}{N},
\qquad
N = b\,\Phi\!\left(\tfrac{b}{se_1}\right) + se_1\!\left(\phi\!\left(\tfrac{b}{se_1}\right)-\phi(0)\right),
$$

where $\Phi,\phi$ are the standard-normal CDF/PDF and $\phi(0)=1/\sqrt{2\pi}$. This is
the predictive density on the observed $y_1$ under the prior "true effect Uniform on
$[0,\infty)$, observed with Gaussian ME `se₁`, renormalized over the observed support
$[0,b]$." Its behavior is:

- near the origin ($y_1\to 0$): $f\approx 0.5/N$ — soft boundary, only half the mass,
  matching the fully-smeared model;
- in the bulk: $f\approx 1/N \approx 1/b$ — the plateau;
- near the cap ($y_1\to b$): $f\approx 1/N$ — **no taper** (the fully-smeared model
  had $0.5/N$ here).

The left/lower half-axes are identical with the sign of the argument flipped so the
"signed" coordinate is positive on the quadrant's own side (`origin - y` for
left/lower, `y - origin` for right/upper). In log-space the code stores four vectors
`lp_y1_left, lp_y1_right, lp_y2_upper, lp_y2_lower`, each $=\ \texttt{std\_normal\_lcdf}(\text{signed}/se)-\log N$.

### 5.4 Assembling the per-pair likelihood

The four quadrant log-densities reuse the correct pair of half-axis terms and the
matching weight (model block, lines 156–161):

$$
\begin{aligned}
\log p_{Q1} &= \log\theta_{\text{disc}} + \texttt{lp\_y1\_left} + \texttt{lp\_y2\_upper} \quad(\text{discordant})\\
\log p_{Q2} &= \log\theta_{\text{conc}} + \texttt{lp\_y1\_right} + \texttt{lp\_y2\_upper} \quad(\text{alleviating})\\
\log p_{Q3} &= \log\theta_{\text{disc}} + \texttt{lp\_y1\_right} + \texttt{lp\_y2\_lower} \quad(\text{discordant})\\
\log p_{Q4} &= \log\theta_{\text{conc}} + \texttt{lp\_y1\_left} + \texttt{lp\_y2\_lower} \quad(\text{aggravating})
\end{aligned}
$$

$$
\log p(y_1,y_2\mid\Theta)=\operatorname{logsumexp}\bigl(\log p_{\text{null}},\ \log p_{Q1},\ \log p_{Q2},\ \log p_{Q3},\ \log p_{Q4}\bigr).
$$

### 5.5 Parameters, weights, and priors

Free parameters: $\mu_1,\mu_2$ (null center), $\sigma_1,\sigma_2>0$ (latent null
scale), $\rho\in(-1,1)$ (shared null correlation), and a 3-simplex `mix_weights`
$=(w_{\text{conc}}, w_{\text{disc}}, w_{\text{null}})$. The concordant weight is split
evenly between Q2 and Q4, and the discordant weight between Q1 and Q3
(transformed-parameters block):

$$
\theta_{\text{conc}}=\tfrac{w_{\text{conc}}}{2},\quad
\theta_{\text{disc}}=\tfrac{w_{\text{disc}}}{2},\quad
\theta_{\text{null}}=w_{\text{null}}.
$$

Priors (model block, lines 130–136), pinned to the paper defaults:

$$
\mu_k\sim N(\mu_\mu{=}0,\ \sigma_\mu{=}2),\qquad
\sigma_k\sim \text{Gamma}(a_\sigma{=}2,\ b_\sigma{=}0.1),
$$
$$
\rho\sim N(0,\ 0.3)\ \text{(weakly informative, gentle pull to 0)},\qquad
\texttt{mix\_weights}\sim\text{Dirichlet}(2,\,2,\,36).
$$

The Dirichlet$(2,2,36)$ prior encodes the expectation that the large majority of pairs
are non-interacting. (The data block also declares `rho_theta` and `kappa_theta`;
these are **unused** by this model and are carried only so the shared data JSON works
for every model variant.)

### 5.6 Fixed run configuration (paper defaults)

Set in [`run_joint_model.py`](../run_joint_model.py) and pinned for
reproducibility:

- **Winsorization:** support clipped per axis to the $[0.10,\ 99.90]$ percentiles
  (`--winsorize-pct 0.10`; output tag suffix `w010`). This concentrates the Uniform/
  half-smeared interaction density so it competes fairly with the null and keeps a
  handful of extreme pairs from stretching the support (see §7).
- **Origin:** fixed at 0 on both axes.
- **Sampling:** 8 chains × (1000 warmup + 1000 sampling), **seed 456**.
- **Inits:** $\mu=0$, $\sigma_k=\text{sd}(y_k)$, $\rho=0.15$,
  `mix_weights = [0.05, 0.05, 0.90]`.

### 5.7 Generated quantities → the reported probabilities

For each pair the `generated quantities` block re-forms the five log-densities and
normalizes by $\log\text{denom}=\log p(y_1,y_2\mid\Theta)$ (lines 197–209):

$$
\begin{aligned}
P(\text{interaction}) &= \exp\bigl(\operatorname{logsumexp}(p_{Q1},p_{Q2},p_{Q3},p_{Q4})-\log\text{denom}\bigr)=1-P(\text{null})\\
P(\text{concordant}) &= \exp\bigl(\operatorname{logsumexp}(p_{Q2},p_{Q4})-\log\text{denom}\bigr)\\
P(\text{discordant}) &= \exp\bigl(\operatorname{logsumexp}(p_{Q1},p_{Q3})-\log\text{denom}\bigr)\\
P(\text{aggravating}) &= \exp(p_{Q4}-\log\text{denom}), \qquad
P(\text{alleviating}) = \exp(p_{Q2}-\log\text{denom})\\
P(\text{no interaction}) &= \exp(p_{\text{null}}-\log\text{denom})
\end{aligned}
$$

`log_lik` $=\log\text{denom}$ is emitted per pair for WAIC/model comparison. The
driver summarizes each probability across posterior draws (posterior mean is the
reported value). The four classes **aggravating, alleviating, discordant,
no-interaction are mutually exclusive and sum to 1**;
`prob_interaction = aggravating + alleviating + discordant` and
`prob_concordant = aggravating + alleviating`.

### 5.8 Merged output schema

[`merge_joint_results.py`](../merge_joint_results.py) joins the
two per-screen summaries with the joint posterior into the 20-column merged table
(one row per unordered ORF pair, `orf1 < orf2`;
`example_data/expected/merged_quad_me_trunc_halfsmeared_toy.tsv`):

```
orf1, orf2, name1, name2,
gi_score_exp1, se_exp1, gi_score_overlaps_zero_exp1,
gi_score_exp2, se_exp2, gi_score_overlaps_zero_exp2,
correlation_exp1, correlation_exp2,
prob_interaction_median_exp1, prob_interaction_median_exp2,
prob_interaction, prob_aggravating, prob_alleviating,
prob_discordant, prob_no_interaction, prob_concordant
```

---

## 6. Signing, thresholding, and the call logic

### 6.1 Calling a pair (`np.select`)

A pair is **called** by whichever directional class clears the threshold `T`. For any
$T\ge 0.5$ at most one class can, since the four class probabilities sum to 1
([`make_hit_matrix.py`](../make_hit_matrix.py); call column in the bundle):

```python
df["call"] = np.select(
    [df.prob_aggravating >= T, df.prob_alleviating >= T, df.prob_discordant >= T],
    ["Negative", "Positive", "Discordant"],
    default="No interaction")
```

Equivalently a pair is a **hit** iff
`max(prob_aggravating, prob_alleviating, prob_discordant) >= T`.

- **Default threshold `T = 0.5`** — the paper's call threshold.
- **`T = 0.95`** — a stricter "confident" cutoff, reported alongside.

Call breakdown over all **290,703** ORF pairs (incl. self-pairs) of the full-library
run:

| Threshold | Negative (aggr) | Positive (allev) | Discordant | Any interaction | No interaction |
|-----------|-----------------|------------------|------------|-----------------|----------------|
| **T = 0.50** | 10,444 (3.59 %) | 2,806 (0.97 %) | 507 (0.17 %) | 13,757 (4.73 %) | 276,946 (95.27 %) |
| **T = 0.95** | 6,480 (2.23 %) | 916 (0.32 %) | 233 (0.08 %) | 7,629 (2.62 %) | 283,074 (97.38 %) |

### 6.2 Signed probability for the heatmap (`signed_maxmag`)

For the diverging-colormap heatmap and the network edges, each pair is collapsed to a
**single signed number**. The paper uses the `signed_maxmag` mode
([`make_signed_prob_matrix.py`](../make_signed_prob_matrix.py)): the **magnitude** is
the full `prob_interaction`, and the **sign** is taken from whichever screen has the
**larger-|GI|** score:

$$
\text{dominant}=\begin{cases} \delta'_{\text{Exp1}} & |\delta'_{\text{Exp1}}|\ge|\delta'_{\text{Exp2}}|\\ \delta'_{\text{Exp2}} & \text{otherwise}\end{cases},
\qquad
\text{signed} = \operatorname{sign}(\text{dominant})\cdot P(\text{interaction}).
$$

For concordant pairs both screens share a sign, so this equals the concordant
direction; for discordant pairs it picks the *stronger* direction (e.g. a
$(-10,+1)$ pair signs **negative**). Aggravating ⇒ negative, alleviating ⇒ positive.
These signed matrices are **not** thresholded, so they are identical at any cutoff.
(Other modes exist — `conc_diff` = $P_{\text{allev}}-P_{\text{aggr}}$;
`signed_interaction` = sign of dominant *class* × $P_{\text{interaction}}$ — but
`signed_maxmag` is the one in use.)

---

## 7. Interpretation: probability is classification confidence, not effect size

The single most important reading rule, and the one added to the manuscript in
response to review:

> **P(interaction) is a classification confidence — how confidently the model assigns
> a pair to the interacting class given *that screen-pair's* noise floor — not a
> continuous measure of interaction strength. For effect size, use the GI score δ′.**

The map from δ′ to P(interaction) is a steep, sigmoid-like transform whose midpoint
sits where |δ′| crosses each screen's noise floor. Because the two screens have
different noise floors (different ATc doses), that midpoint differs between screens
(|δ′| ≈ 2.6 in Exp1 vs ≈ 4.4 in Exp2). A pair whose |δ′| lands between the two
midpoints can be a confident hit in one screen and confidently *not* a hit in the
other — **without** the underlying data disagreeing about direction or magnitude. The
GI scores themselves stay well correlated (R ≈ 0.82) even where the single-screen
probabilities do not (R² ≈ 0.44; Spearman ρ ≈ 0.55; cross-screen ROC AUC ≈ 0.95–0.97).
This is exactly the artifact the joint model is built to absorb: a low probability in
one screen means "this screen's effect was too small relative to its noise floor to
resolve confidently," **not** "there is no interaction."

Practical guidance for readers of the data files:

1. Use `prob_interaction` (or a directional class prob) as a **binary filter** for
   high-confidence hits, not as a quantitative interaction strength.
2. Use **δ′ (`gi_score_exp1/exp2`)** to compare effect sizes across screens/conditions.
3. A low probability in one screen does **not** rule out an interaction; the joint
   model exists to recover such pairs from the combined evidence.

### Worked example — ndhA × ndh (aggravating)

`RVBD0392c` (*ndhA*) × `RVBD1854c` (*ndh*), the two type-II NADH dehydrogenases —
functionally redundant, so knocking down both is far worse than expected (a
classic synthetic-lethal pair):

| Field | Exp1 (100 ng/ml) | Exp2 (500 ng/ml) |
|-------|------------------|------------------|
| GI score δ′ | **−9.67** | **−12.30** |
| SE (`sd_delta_prime_median`) | 0.50 | 0.84 |
| per-screen `prob_interaction_median` | 1.000 | 1.000 |

Both screens place the pair in the **both-negative** quadrant (Q4), with |δ′| far
above either noise floor. The joint model returns:

```
prob_interaction   = 1.00000
prob_aggravating   = 1.00000     ← Q4, both synthetic-sick
prob_alleviating   = 0.0
prob_discordant    = 0.0
prob_no_interaction= 0.0
prob_concordant    = 1.00000
```

Call (any T ∈ {0.5, 0.95}): **Negative** (aggravating). Signed value (`signed_maxmag`):
the larger-|GI| screen is Exp2 (|−12.30| > |−9.67|), which is negative, so
signed = −1.0 × 1.0 = **−1.0** — a strong negative (aggravating) cell in the heatmap.
Biologically, the two type-II NADH dehydrogenases are functionally redundant, so
losing both together is far more deleterious than the additive expectation — the
textbook signature of an aggravating (synthetic-lethal) interaction — and the two
independent screens agree both in **sign** (δ′ < 0) and in **confidence** (P → 1).

---

## Appendix: file map

| File | Role |
|------|------|
| [`joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan`](../joint_normal_uniform_mix_quadrant_me_trunc_halfsmeared.stan) | **primary** joint 5-component half-smeared quadrant model |
| [`univariate_normal_uniform_mix_me.stan`](../univariate_normal_uniform_mix_me.stan) | per-screen 1-D mixture **with measurement error** — the joint model's 1-D analogue; optional `--stan` variant |
| [`joint_normal_uniform_mix.stan`](../joint_normal_uniform_mix.stan) | simple 2-D shared-indicator teaching model (no quadrants, no ME) |
| [`normal_uniform_mix.stan`](../normal_uniform_mix.stan) | **per-screen 1-D Normal/Uniform mixture (default)** → `prob_interaction_median` |
| [`singlepair_splits_delta_twoline_model_v1.stan`](../singlepair_splits_delta_twoline_model_v1.stan) | per-guide-pair two-line fitness model (Y25) |
| [`gi_scoring.py`](../gi_scoring.py), [`logfc_tools.py`](../logfc_tools.py) | fitness fits, Y25, δ′, aggregation |
| [`gam_correction_standalone.R`](../gam_correction_standalone.R) | GAM de-trending of δ′ vs expected fitness |
| [`run_joint_model.py`](../run_joint_model.py) | builds Stan data (winsorize 0.10, seed 456, 8×1000/1000), runs the joint model |
| [`merge_joint_results.py`](../merge_joint_results.py) | assembles the 20-column merged per-pair table |
| [`make_signed_prob_matrix.py`](../make_signed_prob_matrix.py) | `signed_maxmag` gene×gene matrix |
| [`make_hit_matrix.py`](../make_hit_matrix.py) | boolean hit matrix at threshold `T` (call logic) |
