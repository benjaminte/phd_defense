t=0          t_min=0.5   t_emerge=0.7        t_required=3.5        t_0=4.5
 |--------------|------------|-------------------|-------------------|
 formation   habitable    life seen          deadline              now
              |<-- Δt₁ --->|
              |<--------------- Δt₂ ---------------->|

> **Personal recap — not part of the build instructions.**
> The section below is a note to myself, to reconstruct the idea quickly later.
> An agent implementing the figures should skip to *Part A* and ignore this block;
> nothing here is required to produce the plots.

# Recap: how the posterior is actually computed

All numbers below use the **optimistic** parameters: `Δt₁ = 0.2 Gyr`, `Δt₂ = 3.0 Gyr`,
`λ_min = 1e-3 Gyr⁻¹`, `λ_max = 1e3 Gyr⁻¹`.

## Step 1 — The likelihood

One function, no free constants. The ratio form comes from conditioning on our own
existence (`E ⊂ R`, so `P(E|R) = P(E)/P(R)`):

$$\mathcal{L}(\lambda) \equiv P[\mathcal{D}|\lambda]
= \frac{1 - e^{-\lambda \Delta t_1}}{1 - e^{-\lambda \Delta t_2}}
= \frac{1 - e^{-0.2\lambda}}{1 - e^{-3.0\lambda}}$$

Its two limits are the whole evidential content:

$$\mathcal{L} \xrightarrow{\ \lambda \to 0\ } \frac{\Delta t_1}{\Delta t_2} = 0.0667,
\qquad
\mathcal{L} \xrightarrow{\ \lambda \to \infty\ } 1,
\qquad
\mathcal{R} = \frac{\Delta t_2}{\Delta t_1} = 15$$

## Step 2 — The priors

Normalized densities in λ, each on its own support. All three are "uniform" — in a
different variable each time (Jacobian rule `p_λ = p_θ·|dθ/dλ|`):

$$p_\text{U}(\lambda) = \frac{1}{\lambda_\text{max}} = \frac{1}{1000},
\qquad 0 \le \lambda \le 10^{3}$$

$$p_\text{L}(\lambda) = \frac{1}{\lambda \ln(\lambda_\text{max}/\lambda_\text{min})}
= \frac{1}{13.8155\,\lambda},
\qquad 10^{-3} \le \lambda \le 10^{3}$$

$$p_\text{I}(\lambda) = \frac{1}{\lambda^{2}(\lambda_\text{min}^{-1} - \lambda_\text{max}^{-1})}
= \frac{1}{999.999\,\lambda^{2}},
\qquad 10^{-3} \le \lambda \le 10^{3}$$

Each satisfies `∫p dλ = 1` by construction. Only `p_L ∝ 1/λ` is scale-invariant
(equal probability per decade); the other two are informative, in opposite directions.

## Step 3 — The normalization factor (the evidence)

One scalar per prior — the prior-averaged likelihood:

$$P[\mathcal{D}] = \int_{\lambda_\text{min}}^{\lambda_\text{max}}
\mathcal{L}(\lambda)\,p(\lambda)\,d\lambda$$

$$P[\mathcal{D}]_\text{U} = 0.99503, \qquad
P[\mathcal{D}]_\text{L} = 0.45305, \qquad
P[\mathcal{D}]_\text{I} = 0.06771$$

All three lie between the likelihood's plateaus (0.0667 and 1) — an average of
`L` cannot escape `L`'s range. The uniform prior sits at the top because it puts its
mass where `L ≈ 1`; the inverse-uniform sits at the bottom for the mirror reason. Their
ratio, `0.995/0.0677 ≈ 15`, is the Bayes factor again.

## Step 4 — The posterior

$$p(\lambda|\mathcal{D}) = \frac{\mathcal{L}(\lambda)\,p(\lambda)}{P[\mathcal{D}]}$$

written out for the logarithmic prior:

$$p_\text{L}(\lambda|\mathcal{D}) = \frac{1}{0.45305}\cdot
\frac{1 - e^{-0.2\lambda}}{1 - e^{-3.0\lambda}}\cdot\frac{1}{13.8155\,\lambda}$$

## Step 5 — What is actually plotted

Change variable to `u = log₁₀λ`, applying `p(u) = p(λ)·λ·ln10`:

$$p_\text{L}(u|\mathcal{D}) = \frac{\ln 10}{13.8155}\cdot
\frac{\mathcal{L}(10^{u})}{0.45305}
= \frac{1}{6}\cdot\frac{\mathcal{L}(10^{u})}{0.45305}
= 0.36787\,\mathcal{L}(10^{u})$$

The λ's cancel exactly. **Under the logarithmic prior, the plotted posterior is the
likelihood curve, merely rescaled.** Two numerical checks:

| `u` | `L(10^u)` | `0.36787·L` | value in §6 table |
|---|---|---|---|
| +3 | 1.00000 | 0.36787 | 3.679e-01 ✓ |
| −3 | 0.06676 | 0.024559 | 2.456e-02 ✓ |

Note `ln10/13.8155 = 1/6` exactly — the flat log prior in `u`, reappearing.

## Why this is the paper's whole argument

Under the only non-informative prior, the posterior *is* the likelihood. So whatever
shape the data have is the entire shape of the answer — and that shape is two flat
plateaus a factor of 15 apart. Meanwhile the three "ignorance" priors disagree among
themselves about `P(λ < 1 Gyr⁻¹)` by a factor of ~1000 (0.001 / 0.50 / 0.999). A factor
of 15 cannot overturn a factor of 1000, so the prior decides and the data only nudge.

## Where `P[D]` lives in the code

It never appears as a named variable, because the script never normalizes the prior
separately. The single generic helper

```python
def density_in_u(p_lambda, lam, u):
    p_u = p_lambda * lam * np.log(10)
    return p_u / np.trapezoid(p_u, u)      # <- P[D] is this denominator
```

is called with `p_lambda = prior_shape * L`, where `prior_shape` is the bare `1`,
`1/λ` or `1/λ²`. Writing the true prior as `p = c · prior_shape`, the same unknown `c`
appears in numerator and denominator and cancels, along with `P[D]`. One division does
both jobs, which is why the same function works for all six curves.

To inspect `P[D]` as a number, normalize the prior first:

```python
prior_u = density_in_u(prior_shape, lam, u)   # now integrates to 1
Z       = np.trapezoid(prior_u * L, u)        # this is P[D], on its own
post_u  = prior_u * L / Z
```

Both routes give identical posteriors.

---
---

# Recreating Figures 1 and 3 of Spiegel & Turner (2011)

This document has two blocks:

- **Part A — Figure 1** (§1–§8): PDF and CDF of λ under three priors. Establishes the
  model, the likelihood, the priors, the log-abscissa convention and the numerical
  machinery. Read this first; Part B depends on all of it.
- **Part B — Figure 3** (§9–§14): CDF of λ with and without an independent second
  abiogenesis, under the logarithmic prior only.

---

# Part A — Figure 1

**Task for the implementing agent.** Write a single self-contained Python script
(`numpy` + `matplotlib` only) that reproduces both panels of Figure 1 from:

> D. S. Spiegel & E. L. Turner, *Bayesian analysis of the astrobiological implications
> of life's early emergence on Earth*, PNAS **109**(2), 395–400 (2012);
> doi:10.1073/pnas.1111694108.

The figure shows the prior and posterior probability of the abiogenesis rate λ under
three different priors. **Left panel:** probability density functions (PDFs), log
ordinate. **Right panel:** cumulative distribution functions (CDFs), linear ordinate.
Both panels use `log10(λ)` as the abscissa.

Do not proceed to plotting until the validation numbers in §6 are reproduced. Several
of them are diagnostic of the two mistakes described in §5, which are easy to make and
produce a plot that looks plausible but is wrong.

---

## 1. The model

λ is the rate of abiogenesis (units Gyr⁻¹) on a sterile but habitable planet. Life
arises as a Poisson process, so the probability that life has arisen within a window
of length Δt is

```
P_life(Δt) = 1 - exp(-λ Δt)
```

Two windows matter, both measured from `t_min`, the time at which the planet first
became habitable:

| Symbol | Definition | Meaning |
|---|---|---|
| `Δt₁` | `t_emerge - t_min` | window within which life demonstrably appeared |
| `Δt₂` | `t_required - t_min` | window within which life *had to* appear for observers to exist now |

where `t_required = min(t₀ - δt_evolve, t_max)`, with `t₀` the planet's current age,
`δt_evolve` the time from first life to intelligent observers, and `t_max` the last time
abiogenesis remains possible.

### The likelihood (paper's Eq. 3)

Because we can only observe our own history if abiogenesis occurred within `Δt₂`, the
likelihood is a *conditional* probability. Writing `E` = "life arose within Δt₁" and
`R` = "life arose within Δt₂", and noting `E ⊂ R`:

```
P[D|M] = P(E|R) = P(E)/P(R)
```

```
                1 - exp(-λ Δt₁)
    P[D|M]  =  ─────────────────                                  (Eq. 3)
                1 - exp(-λ Δt₂)
```

This is the only place the data enter. Its two limits are:

- `λ → 0`  :  `P[D|M] → Δt₁/Δt₂`   (λ cancels — the likelihood goes flat)
- `λ → ∞`  :  `P[D|M] → 1`

so the whole evidential content of the observation is the ratio of the two plateaus,
the Bayes factor `R = Δt₂/Δt₁`.

### Model parameters for Figure 1

Figure 1 uses the **"optimistic"** column of the paper's Table 1. All times in Gyr:

```
t_min     = 0.5
t_emerge  = 0.7
t_max     = 10.0
δt_evolve = 1.0
t₀        = 4.5
```

giving `t_required = 3.5`, `Δt₁ = 0.2`, `Δt₂ = 3.0`, and Bayes factor `R = 15`.

Hard-code `Δt₁ = 0.2` and `Δt₂ = 3.0` only as a fallback — prefer deriving them from
the primitive parameters above, so that the other Table 1 columns can be swapped in:

| | Hypothetical | Conservative 1 | Conservative 2 | **Optimistic** |
|---|---|---|---|---|
| `t_emerge` | 0.51 | 1.3 | 1.3 | **0.7** |
| `t_max` | 10 | 1.4 | 10 | **10** |
| `δt_evolve` | 1 | 2 | 3.1 | **1** |
| `Δt₁` | 0.01 | 0.80 | 0.80 | **0.20** |
| `Δt₂` | 3.00 | 0.90 | 0.90 | **3.00** |
| `R` | 300 | 1.1 | 1.1 | **15** |

---

## 2. The three priors

Prior bounds for Figure 1: `λ_min = 1e-3 Gyr⁻¹`, `λ_max = 1e3 Gyr⁻¹`.

Each prior is "uniform" in a different variable. Converting to a density in λ uses the
Jacobian rule `p_λ(λ) = p_θ(θ)·|dθ/dλ|`:

| Name | Flat in | Density in λ | Normalized form |
|---|---|---|---|
| `Uniform` | λ | `∝ 1` | `1/λ_max` on `[0, λ_max]` |
| `Log` | `log₁₀ λ` | `∝ 1/λ` | `1 / (λ · ln(λ_max/λ_min))` on `[λ_min, λ_max]` |
| `InvUnif` | `1/λ` | `∝ 1/λ²` | `1 / (λ² · (λ_min⁻¹ - λ_max⁻¹))` on `[λ_min, λ_max]` |

Note the `Uniform` prior's support starts at 0, not at `λ_min` (this follows the paper's
caption). The probability mass it places below `λ_min` is `1e-6` and is invisible on the
plot, so it is safe to normalize numerically over the plotted range; do not treat a
`1e-6` discrepancy in its CDF as an error.

In the script it is sufficient to define the three priors up to proportionality as
`1`, `1/λ`, `1/λ²` and normalize numerically (§4, step 5) — the analytic constants above
are given for checking.

---

## 3. Why the abscissa is `log₁₀ λ`, and what it forces

**The axis.** λ plausibly spans many orders of magnitude (the paper's Fig. 2 extends to
`1e-22`). On a linear axis the entire region of interest (`λ < 1`) collapses to a single
pixel. The question being asked is about the *order of magnitude* of λ, so the natural
coordinate is `u = log₁₀ λ`, where equal distances are equal ratios.

**The consequence — this is the step most often missed.** A probability density is
defined *per unit of its variable*. Once the abscissa is `u`, the ordinate must be the
density per unit `u`, not per unit λ:

```
p(u) = p(λ) · |dλ/du| = p(λ) · λ · ln(10)
```

The factor `λ·ln(10)` is mandatory. With it, area under the plotted curve is genuine
probability, and the `Log` prior appears as a horizontal line while the `Uniform` prior
appears as a line of slope **+1** and the `InvUnif` prior as slope **−1**. Without it,
the `Log` prior would wrongly appear to fall with slope −1 and the `Uniform` prior would
wrongly appear flat — i.e. the visual argument of the figure inverts.

---

## 4. Algorithm

1. **Set parameters.** `t_min`, `t_emerge`, `t_max`, `δt_evolve`, `t₀` as in §1; derive
   `t_required`, `Δt₁`, `Δt₂`. Set `λ_min = 1e-3`, `λ_max = 1e3`.

2. **Build a grid uniform in `u`, not in λ.**
   ```python
   u   = np.linspace(-3, 3, 4001)     # ≥2001 points; 4001 is comfortable
   lam = 10.0**u
   ```
   Sampling uniformly in λ instead would place ~99.9 % of the nodes in the top decade
   and leave the `λ < 1` region — the region the paper's conclusion is about —
   resolved by a handful of points.

3. **Evaluate the likelihood** (Eq. 3) on the grid. Use `expm1` (see §5).

4. **Evaluate the three priors** as densities in λ, up to proportionality:
   `np.ones_like(lam)`, `1/lam`, `1/lam**2`.

5. **Form and normalize six curves.** Three priors and three posteriors
   (`posterior ∝ prior × likelihood`). For each, convert to a density in `u` and
   normalize so it integrates to 1 over the grid:
   ```python
   def density_in_u(p_lambda, lam, u):
       p_u = p_lambda * lam * np.log(10)      # Jacobian
       return p_u / np.trapezoid(p_u, u)      # np.trapz on numpy < 2.0
   ```
   Normalizing *after* the Jacobian conversion, by integrating over `u`, is the
   simplest correct order; normalizing in λ and then converting works too and must give
   the same answer (a useful self-check).

6. **Compute the CDFs** by cumulative integration of the `u`-densities:
   ```python
   def cdf(p_u, u):
       inc = 0.5 * (p_u[1:] + p_u[:-1]) * np.diff(u)   # trapezoid increments
       return np.concatenate([[0.0], np.cumsum(inc)])
   ```
   Each CDF must start at 0 at `u = -3` and end at 1 at `u = +3`. The paper describes
   the ordinate as "the integrated probability from 0 to the abscissa", which for the
   two priors supported on `[λ_min, λ_max]` is the same thing.

7. **Plot.** Two side-by-side axes sharing the abscissa; styling in §7.

---

## 5. Numerical pitfalls

**(a) Catastrophic cancellation in `1 - exp(-x)`.** Compute it as `-np.expm1(-x)`, never
as `1 - np.exp(-x)`. For the Figure 1 range the naive form is merely inaccurate, but if
the script is later extended to the paper's Fig. 2 (`λ_min = 1e-22`), both numerator and
denominator evaluate to exactly `0.0` in double precision and the likelihood becomes
`nan`. With `expm1` the ratio remains accurate down to arbitrarily small λ.

**(b) Forgetting the Jacobian.** See §3. The diagnostic is the `Log` prior: on the
finished plot it must be a horizontal line at exactly `1/6` (one over the number of
decades). If it slopes, the Jacobian is missing.

**(c) Sampling the grid in λ.** See §4, step 2. The diagnostic is the `Log` prior's CDF,
which must be a straight diagonal from (−3, 0) to (+3, 1). If it is curved, the grid is
wrong.

**(d) Ordinate label.** The left panel's ordinate is a density *per decade*, not per
Gyr⁻¹. The paper labels it simply "Probability Density"; matching that is fine, but do
not label it "per Gyr⁻¹".

---

## 6. Validation targets

Reproduce these before plotting. Print them; do not just eyeball the figure.

**Likelihood plateaus**
```
L(λ=1e-3) = 0.06676   (≈ Δt₁/Δt₂ = 0.066667)
L(λ=1e3)  = 1.000000
Bayes factor Δt₂/Δt₁ = 15.0
```

**Left panel — density per decade `p(u)`**

| curve | `u=-3` | `u=0` | `u=+3` |
|---|---|---|---|
| Uniform prior | 2.303e-06 | 2.303e-03 | 2.303e+00 |
| Uniform posterior | 1.545e-07 | 4.414e-04 | 2.314e+00 |
| Log prior | 1.667e-01 | 1.667e-01 | 1.667e-01 |
| Log posterior | 2.456e-02 | 7.018e-02 | 3.679e-01 |
| InvUnif prior | 2.303e+00 | 2.303e-03 | 2.303e-06 |
| InvUnif posterior | 2.270e+00 | 6.488e-03 | 3.401e-05 |

Two structural checks on this table: the `Log` prior is constant at `1/6 = 0.1667`, and
the `Uniform` and `InvUnif` priors are exact mirror images of one another.

A third check ties the plot to the physics: the `Log` posterior's plateau ratio is
`0.3679 / 0.02456 = 15.0`, i.e. the Bayes factor read directly off the left panel as the
height of the step.

**Right panel — CDF**

| curve | `u=-2` | `u=0` | `u=+2` |
|---|---|---|---|
| Uniform prior | 0.0000 | 0.0010 | 0.0999 |
| Uniform posterior | 0.0000 | 0.0001 | 0.0954 |
| Log prior | 0.1667 | 0.5000 | 0.8332 |
| Log posterior | 0.0247 | 0.0912 | 0.6319 |
| InvUnif prior | 0.9001 | 0.9990 | 1.0000 |
| InvUnif posterior | 0.8895 | 0.9937 | 0.9999 |

Tolerance: 3 significant figures. Small deviations in the last digit from grid
resolution are acceptable; a factor-of-λ deviation anywhere means the Jacobian is wrong.

**Qualitative acceptance criteria** (these are the point of the figure):

- In the **left** panel all three posteriors show a similar step of roughly an order of
  magnitude near `λ ~ 0.5 Gyr⁻¹`, which makes the data *look* influential.
- In the **right** panel the `Uniform` and `InvUnif` posterior CDFs lie essentially on
  top of their own priors — the data are nearly irrelevant to the integrated
  conclusion — while the `Log` posterior separates clearly from its prior.
- The `Log` posterior CDF at `λ = 1 Gyr⁻¹` (`u = 0`) is ≈ 0.09. The paper quotes ≈ 12 %
  for the posterior probability that `λ < 1 Gyr⁻¹`; a value in the 0.09–0.12 range is
  correct, the residual difference coming from grid extent and the treatment of the
  prior edges. Do not tune parameters to force exactly 0.12.

---

## 7. Plot specification

**Layout.** `plt.subplots(1, 2, figsize=(13, 5.7))`. Both panels titled `Optimistic`.

**Abscissa (both panels).** `u` from −3 to +3, label:
`$\log_{10}[\lambda]$   ($\lambda$ in Gyr$^{-1}$)`

**Left panel.** `set_yscale('log')`, limits `1e-7` to `1e1`, ordinate label
`Probability Density`.

**Right panel.** Linear ordinate, limits `0` to `1`, label `Cumulative Probability`.

**Curves.** Six per panel — one prior and one posterior for each of the three priors.

| Series | Style | Colour used in the reproduction |
|---|---|---|
| prior | dashed, lw ≈ 1.6 | — |
| posterior | solid, lw ≈ 2.2 | — |
| Uniform | | green `#3ddc4a` |
| Log | | purple `#b48ce8` |
| InvUnif | | red/orange `#ff6a4d` |

The published caption calls the logarithmic curves "blue"; the figure as printed renders
them purple. Either is acceptable — keep the three colours clearly distinct.

**Annotation.** In the left panel, the three series names in their own colours, plus two
lines reading `Posterior: Solid` and `Prior: Dashed`. The original uses a monospace font
on a dark background (`#2b2f3a`) with white axes, ticks and labels; a conventional white
background is equally acceptable unless matching the printed figure exactly is required.

**Output.** Save as PNG at `dpi >= 150`, and print the §6 validation tables to stdout.

---

## 8. Optional extensions

These reuse the same machinery and are worth supporting via parameters rather than
separate scripts:

- **Other Table 1 models.** Swap in the Conservative columns (`Δt₁ = 0.80`,
  `Δt₂ = 0.90`). The `Log` prior and posterior should become nearly indistinguishable,
  since `R = 1.1`.
- **Paper's Fig. 2.** Lower `λ_min` towards `1e-22` and plot the posterior median and
  2σ lower bound against `λ_min`. The median is roughly stable; the lower bound tracks
  `λ_min` without limit. This is the case that *requires* `expm1` (§5a).
- **Paper's Fig. 3.** Independent abiogenesis — specified in full in Part B below.

---

# Part B — Figure 3

**Task.** Extend the Part A script (or write a second one sharing the same helper
functions) to reproduce Figure 3: the cumulative distribution of λ under the
**logarithmic prior only**, comparing the posterior with and without the discovery of a
second, independent origin of life.

Everything in §1–§5 of Part A still applies unchanged — same Poisson model, same
`u = log₁₀ λ` abscissa, same Jacobian, same `expm1` requirement, same trapezoidal CDF.
Only the likelihood changes, and only one prior is used.

## 9. The physical argument (why the likelihood changes shape)

Earth's datum is subject to a selection effect: we had to find ourselves on a planet
where abiogenesis occurred early enough to produce us. That is what the denominator of
Eq. 3 encodes, and it is what makes the likelihood go *flat* as `λ → 0`.

We did **not** have to find ourselves in a system where life arose a second, independent
time. A second sample is therefore not subject to that selection effect, and its
contribution to the likelihood carries **no denominator**.

This is the whole point of the figure, and it is visible in the code as the asymmetry
between the two factors below. An implementation that conditions the Mars factor the
same way as the Earth factor will produce a plot that looks similar but destroys the
result.

## 10. The likelihood (paper's Eq. 7)

```
             ┌                                  ┐   1 - exp(-λ Δt₁_Earth)
  P[D|M] =   │ 1 - exp(-λ Δt_Mars)              │ · ─────────────────────      (Eq. 7)
             └                                  ┘   1 - exp(-λ Δt₂_Earth)
        ╰──────── new, unconditioned ────────╯   ╰──── Eq. 3, unchanged ────╯
```

The two events are assumed independent (no panspermia in either direction) and to share
a single λ, so their probabilities multiply.

Limits, to be checked against §13:

- `λ → ∞` : both factors → 1, so `P[D|M] → 1`, same as before.
- `λ → 0` : the Earth factor → `Δt₁/Δt₂` as before, but the new factor → `λ·Δt_Mars`,
  which goes to **zero linearly**. The small-λ plateau is gone.

Consequence: the likelihood no longer has a bounded step of `R = Δt₂/Δt₁ = 15`. It
suppresses small λ without limit, so the posterior stops being a rescaled copy of the
prior in that region. This is the structural difference the figure exists to show.

## 11. Parameters

Earth: the **optimistic** model of Part A, unchanged.

```
Δt₁ = t_emerge - t_min = 0.7 - 0.5 = 0.2 Gyr
Δt₂ = t_required - t_min = 3.5 - 0.5 = 3.0 Gyr
```

Mars (from the paper's text, not from Table 1):

```
t_min^Mars    = 0.5 Gyr
t_emerge^Mars = 1.0 Gyr        (also taken as t_max^Mars)
Δt_Mars       = 0.5 Gyr
```

Prior: logarithmic only, `λ_min = 1e-3 Gyr⁻¹`, `λ_max = 1e3 Gyr⁻¹`, i.e. `p(λ) ∝ 1/λ`.

Note that `Δt_Mars` is a single window with no partner to divide by. Do not attempt to
construct a `Δt₂` for Mars — there is none, by the argument of §9.

## 12. Algorithm

Identical to §4 of Part A, with these changes:

1. Same grid: `u = np.linspace(-3, 3, 4001)`, `lam = 10.0**u`.
2. Compute **two** likelihoods on that grid:
   ```python
   L_earth = (-np.expm1(-lam*dt1)) / (-np.expm1(-lam*dt2))
   L_indep = (-np.expm1(-lam*dt_mars)) * L_earth
   ```
3. Use the single prior `prior = 1/lam`.
4. Form **three** curves: `prior`, `prior * L_earth`, `prior * L_indep`.
5. Convert each to a density in `u` (Jacobian `λ·ln10`), normalize, then integrate to a
   CDF — reuse the Part A helpers verbatim.
6. Plot all three CDFs on one axis.

## 13. Validation targets

**Likelihood values** (check the small-λ behaviour especially — it is the whole point):

| `u = log₁₀λ` | `L_earth` | `L_indep` | `λ·Δt_Mars` |
|---|---|---|---|
| −3 | 6.676e-02 | 3.337e-05 | 5.000e-04 |
| −2 | 6.761e-02 | 3.376e-04 | 5.000e-03 |
| −1 | 7.639e-02 | 3.721e-03 | 5.000e-02 |
| 0 | 1.908e-01 | 7.506e-02 | 5.000e-01 |
| +1 | 8.650e-01 | 8.592e-01 | 5.000e+00 |
| +3 | 1.000000 | 1.000000 | 5.000e+02 |

Two structural checks:

- `L_earth` is flat across `u = -3 … -1` (values within 15 % of `Δt₁/Δt₂ = 0.0667`),
  whereas `L_indep` falls by a factor of ~10 per decade over the same range. That
  contrast *is* the result.
- At `u = -3`, `L_indep / L_earth = 5.00e-04`, which must equal `λ·Δt_Mars` to three
  digits. If it does not, the new factor has been conditioned or mis-scaled.

**CDF table**

| `u` | prior | posterior (Earth only) | posterior (independent) |
|---|---|---|---|
| −3 | 0.0000 | 0.0000 | 0.0000 |
| −2 | 0.1667 | 0.0247 | 0.0001 |
| −1 | 0.3333 | 0.0505 | 0.0006 |
| 0 | 0.5000 | 0.0912 | 0.0097 |
| +1 | 0.6667 | 0.2722 | 0.1801 |
| +2 | 0.8332 | 0.6319 | 0.5853 |
| +3 | 1.0000 | 1.0000 | 1.0000 |

The headline number: `P(λ < 1 Gyr⁻¹)` falls from **0.50** (prior) to **0.09** (Earth
alone) to **0.010** (with an independent second origin) — roughly another factor of 9
beyond what all of Earth's history achieves. The Earth-only column must agree with the
`Log` posterior column of Part A §6; if it does not, the two scripts have diverged.

**Percentiles** (useful as a second, independent check on the CDFs):

| curve | median `u` | 2σ (95.4 %) lower bound `u` |
|---|---|---|
| prior | 0.000 | −2.726 |
| posterior, Earth only | +1.641 | −1.181 |
| posterior, independent | +1.794 | **+0.472** |

The medians barely move between the two posteriors. The 2σ lower bound moves by more
than 1.6 decades and, crucially, crosses from below `λ = 1` to above it. That is the
quantitative statement of "a second sample forecloses on very rare", and it is a better
acceptance test than the median.

**Qualitative acceptance criteria:**

- The prior is a straight diagonal from (−3, 0) to (+3, 1).
- The Earth-only posterior lies below the prior everywhere, but remains visibly
  *above zero* across the whole left half of the plot — it retains a small-λ tail.
- The independent-abiogenesis posterior is pressed essentially flat against zero for
  `u < 0`, then rises steeply. The disappearance of that left-hand tail is the figure's
  message.
- The two posteriors converge at the right-hand edge.

## 14. Plot specification

**Layout.** Single axis, roughly `figsize=(7.5, 6)`.

**Abscissa.** `u` from −3 to +3, label
`$\log_{10}[\lambda]$   ($\lambda$ in Gyr$^{-1}$)`.

**Ordinate.** Linear, 0 to 1, ticks every 0.1, label `Cumulative Probability`.

**Curves and legend.** Three entries, legend titled `Independent Life, log. prior`,
placed upper-left:

| Legend entry | Curve | Style |
|---|---|---|
| `Prior` | logarithmic prior CDF | dashed |
| `Posterior if independent abiogenesis` | `prior × L_indep` | solid |
| `Posterior for Earth (Optimistic)` | `prior × L_earth` | solid |

Keep the three colours distinct; the Part A palette (`#b48ce8`, `#3ddc4a`, `#ff6a4d`)
works. Same dark-background convention as Part A, or a white background if the printed
styling is not required.

**Output.** Save as PNG at `dpi >= 150` and print both §13 tables to stdout.
