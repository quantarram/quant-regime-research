# The Ceiling Holds: Testing Five AI/ML Architectures Against an Empirically Measured Predictability Limit

Draft preprint — CPE research series, Paper 14

By Arun Ramanathan

---

## Abstract

A long-standing assumption in applied machine learning holds, at least implicitly, that with enough data and enough computational power, a sufficiently capable model can approximate almost any target arbitrarily well. Chaos theory and stochastic dynamics overturned the equivalent classical intuition in the physical sciences decades ago: some systems have a genuine, structural predictability limit that no increase in data volume or model sophistication can overcome, because the limit comes from the dynamics themselves — sensitive dependence on initial conditions, or genuine stochastic forcing — not from any deficiency correctable by scale. This research program has spent its last several papers measuring exactly such a limit for financial instruments, entirely empirically and without reference to any specific model (Paper 11), and turning it into a practical training-window prescription (Paper 13). This paper asks the natural next question in two parts. First: within that measured limit, does giving five architecturally very different models — climatology, an unregularized decision tree, a reinforcement-learning policy-gradient forecaster, a conditional generative adversarial network, and a conditional variational autoencoder — an identical, fair, non-stale training budget produce any real separation in skill, or does the ceiling bind so tightly that architectural sophistication buys nothing over the simplest possible model? Second, and directly modeled on how Lorenz's theoretical predictability limit was later confirmed not by argument but by actual numerical weather models empirically losing skill past the same horizon: does training the same five architectures on progressively larger windows that extend past that measured limit cause a visible, shared degradation in all of them, arriving at roughly the point Paper 11 already located by a completely model-free method? Both answers are shown here entirely as figures — predicted price against actual price, nothing more — per this research program's standing view that this kind of question is more honestly settled by direct visual demonstration than by a significance test. Within the limit, no architecture beats climatology; two of the four non-trivial models (the reinforcement-learning forecaster and the GAN) converge almost exactly onto it. Trained on windows extending well past the limit, the same models drift into a slow, visible detachment from actual subsequent price behavior — not a sudden cliff, but a build that becomes unmistakable by four to eight times the measured limit, in every instrument tested. The predictability limit does not just describe market structure, as Paper 11 first showed, or prescribe a training window, as Paper 13 then showed. It is also, this paper argues, a genuine ceiling that architecture cannot buy its way past — confirmed here the same way Lorenz's own limit eventually was: not by theory alone, but by watching real models hit it.

## Plain Language Summary

A common, mostly unstated assumption behind a lot of modern AI/ML work is that more data and a more powerful model can eventually predict almost anything. In the physical sciences this was tested and found wanting: chaos theory showed that some systems, like weather, have a hard limit on how far ahead they can be predicted no matter how good your model or how much data you feed it, because the limit lives in the system's own dynamics, not in the model. This research program has already measured a limit like that for financial markets, directly and empirically, with no model involved in the measurement itself. This paper runs two tests of what that limit actually means for AI/ML in practice. First, we give five very different kinds of models — a plain average, an intentionally overfit decision tree, a reinforcement-learning agent, a generative adversarial network, and a variational autoencoder — the exact same fair amount of training data, sized to stay safely within that measured limit, and simply watch which one predicts real subsequent prices best. Second, we take those same five models and start feeding them progressively more training data, extending further and further past that limit, the way a real team chasing "more data" often does, and watch what happens. In the first test, nothing beats the plain average; two of the fancier models essentially become the plain average. In the second test, every model — regardless of how it works internally — starts drifting away from reality once its training data reaches well past the measured limit, and the drift gets worse the further past it they go. This is the same pattern atmospheric scientists saw decades ago: a theoretical predictability limit, discovered before any model could test it, that real forecasting systems later confirmed simply by running into it. We show this entirely as plots of predicted price against real price — no test scores, no p-values — because for a question like this, seeing it directly is the more honest form of evidence.

---

## 1. Introduction

A version of the following claim is close to an article of faith in much of applied AI/ML: given enough data, and a powerful enough model, almost any target can be predicted arbitrarily well. It is easy to see why the claim is appealing — it is, in essence, a restatement of Laplace's determinism, the idea that a sufficiently complete description of a system's state, combined with a sufficiently powerful computation, renders its future fully knowable. The physical sciences already ran this experiment and got a different answer. Chaos theory — beginning with Lorenz's discovery that a deterministic system can exhibit sensitive dependence on initial conditions, and continuing through the broader study of genuinely stochastic dynamics — showed that some systems have a predictability horizon that is structural, not a matter of insufficient data or insufficient computing power. Past that horizon, a model does not get worse because it is a bad model; it gets worse because the information needed to do better has, in a real and irreducible sense, already decayed away.

This research program's own arc has been building toward exactly this point in financial markets specifically. Paper 11 measured a predictability limit for a sample of instruments entirely empirically, with no model of any kind in the loop — a non-parametric structure-function decomposition of an instrument's own price dynamics, nothing more. Paper 13 turned that number into a working prescription: train on about half of it, and treat data from well past it as belonging to a different, no-longer-current regime. Both papers established that the limit exists and that respecting it improves a deliberately simple diagnostic model's tracking of subsequent reality. Neither paper, however, directly tested the claim this paper opens with: that a sufficiently sophisticated and data-hungry architecture might simply predict past the limit anyway. This paper runs that test, in two complementary forms.

**Experiment 1** asks whether, given the exact same fair training budget — sized to stay within the measured limit, identical for every model, so that none is quietly cheating by seeing more (or staler) data than another — architectural sophistication produces any separation in skill at all. Climatology (Paper 11's own respected baseline throughout this research program), an unregularized decision tree (Paper 13's diagnostic), a reinforcement-learning policy-gradient forecaster, a conditional generative adversarial network, and a conditional variational autoencoder are all given the identical half-window training budget and compared directly, side by side, against actual subsequent price.

**Experiment 2** asks the historically-minded question directly: if the predictability limit is genuinely structural rather than a property of any one model, then training data crossing it should degrade any architecture's forecasts, regardless of how that architecture works internally — the same way Lorenz's theoretical predictability limit, derived from the dynamics of the atmosphere itself with no numerical model involved, was later independently confirmed once actual numerical weather prediction models were built and run: they, too, lost forecast skill precipitously past almost exactly the horizon the theory had already predicted, and they did so regardless of which specific model was used. This paper's second experiment sweeps the same five architectures' training-window size from well within the measured limit out to eight times beyond it, holding the test segments being predicted fixed throughout, and watches whether all five degrade in roughly the same place.

**A note on scope**, carried over unchanged from Paper 13: nothing about the underlying question is specific to financial markets. A predictability limit that no architecture can buy its way past is, if real, a property of any non-stationary system with genuine structural limits on how far its own dynamics remain self-similar — weather and climate models chief among them, since that is where this idea originated, but also physical and biological simulation surrogates, industrial sensor systems, epidemiological forecasting, and any other domain where "just add more data and a bigger model" is treated as a default strategy rather than a question to be tested against a measured ceiling. This paper's tests are run in finance because that is where this research program's own predictability-limit machinery already exists; the claim being tested is not bounded by that domain.

**A note on method**, also carried over unchanged: this paper reports no null-hypothesis test, no resampling procedure, and no significance score of any kind, in either experiment. Both are shown entirely as direct, visual comparisons of predicted price against actual price. This is a deliberate, standing choice of evidentiary standard for this research program, not an omission — whether a model's predictions track reality, or drift away from it as training data grows stale, is something a reader can judge directly from the curves themselves, and this paper holds that direct judgment is the more honest and more decisive standard for a question this practical.

## 2. Background: The Measured Limit This Paper Tests Against

Paper 11 defines, for an instrument's own return series, the correlated–decorrelated structure-function gap at lag τ and moment order q,

<div class="eqn-row"><span class="eqn-spacer"></span><img src="predictor_v1/eq13_figs/eq1_gap.svg" alt="Equation 1"><span class="eqn-num">(1)</span></div>

computed as empirical sample moments directly — no power-law fit, no scaling exponent, at any point. The predictability limit τ\* is the lag at which this gap peaks among tradeable lags, at moment order q = 2:

<div class="eqn-row"><span class="eqn-spacer"></span><img src="predictor_v1/eq13_figs/eq2_predictability_limit.svg" alt="Equation 2"><span class="eqn-num">(2)</span></div>

Paper 13 showed that a training window sized to w = ⌊τ\*/2⌋, applied to the immediately following window of the same size, keeps every training-test pair within τ\* days of each other — the correct design, since the naive choice of w = τ\* for both windows allows train-test gaps of nearly 2τ\*, already violating the limit it is meant to respect. This paper reads τ\* directly from Paper 11's already-published results for the same 12 instruments Paper 13 used (`predictability_paper/results_correlated_decorrelated.json`); no new estimation of the limit itself is performed anywhere in this paper.

## 3. Experiment 1: An Architecture Bake-off Within the Predictability Limit

### 3.1 Design

Every architecture in this experiment receives the identical training budget: the w = ⌊τ\*/2⌋ days immediately preceding each test segment, walked forward across an instrument's full available history, exactly as in Paper 13. No architecture is ever shown more data, staler data, or differently-scoped data than any other — the only variable being tested is what each architecture does with the same fair budget. Because a fair test of a data-hungry architecture still requires the training budget to contain as many real rows as possible, this experiment is restricted to the three of Paper 13's twelve instruments with the largest measured predictability limits: MSFT (τ\* = 63d, w = 31d), EURUSD=X (τ\* = 43d, w = 21d), and XLF (τ\* = 33d, w = 16d).

### 3.2 Five competing models

**Climatology** predicts the training window's own mean forward return — no covariates, no learning of any kind. Per this research program's standing position, climatology is treated as a genuine competing model here, not a strawman: if it wins, that is a real result, not a failure to be explained away.

**The overfit tree** is Paper 13's own diagnostic, an unconstrained-depth decision tree (`DecisionTreeRegressor`, `max_depth=None, min_samples_leaf=1, min_samples_split=2`; Breiman et al., 1984) fit on the instrument's ten most recent lagged daily returns, with zero regularization.

**The reinforcement-learning forecaster** treats prediction as a continuous-action bandit problem: a Gaussian policy over the predicted return, trained by policy-gradient ascent on a reward equal to the negative squared prediction error, with a running reward baseline (Williams, 1992):

<div class="eqn-row"><span class="eqn-spacer"></span><img src="predictor_v1/eq14_figs/eq1_rl_policy.svg" alt="Equation 3"><span class="eqn-num">(3)</span></div>

**The conditional GAN** pits a generator, producing a candidate forward return conditioned on the same lagged-return features plus a latent noise draw, against a discriminator trained to distinguish real (feature, realized-return) pairs from generated ones — trained adversarially in the standard minimax form (Goodfellow et al., 2014):

<div class="eqn-row"><span class="eqn-spacer"></span><img src="predictor_v1/eq14_figs/eq2_gan_minimax.svg" alt="Equation 4"><span class="eqn-num">(4)</span></div>

with the forecast at test time taken as the mean of 200 generator draws.

**The conditional VAE** learns an encoder mapping (features, realized return) to a latent distribution and a decoder mapping (features, latent draw) back to a predicted return, trained on the evidence lower bound with the reparameterization trick (Kingma & Welling, 2014):

<div class="eqn-row"><span class="eqn-spacer"></span><img src="predictor_v1/eq14_figs/eq3_vae_elbo.svg" alt="Equation 5"><span class="eqn-num">(5)</span></div>

with the forecast taken as the mean of 200 decoder draws from the prior at test time.

**A note on scale, stated plainly.** All three of these architectures are implemented from scratch in numpy, with linear (not deep) generators, discriminators, encoders, decoders, and policies, and trained with manual gradients — no deep learning framework was used. This is not a limitation applied reluctantly; it is the appropriate scale for the problem this experiment actually poses. A training window of 16 to 31 rows cannot justify a deep network under any architecture, and forcing one in would not make the resulting comparison more informative — it would just make it a comparison of which architecture overfits a tiny sample least gracefully. These are genuine, correctly-trained instances of each architectural family, scaled to the sample size the predictability limit itself dictates, which is precisely the point: the predictability limit does not just bound how far ahead a model can see, it also bounds how much training data any architecture — however sophisticated in principle — actually has to work with.

## 4. Experiment 1 Results

Figures 1–3 show, for each instrument, actual price (black) against every model's forecast price, walked forward across full available history (top panel, log scale) and zoomed to the most recent two years (bottom panel, linear scale, where the separation between models is easiest to see by eye). Climatology is drawn last and dashed specifically so it remains visible underneath the other curves — the point of including it is to be able to see directly when a more sophisticated model's curve sits on top of it.

![Figure 1](predictor_v1/65_architecture_bakeoff_MSFT.png)
*Figure 1. MSFT. In the recent-period panel, the RL forecaster and the conditional GAN track almost exactly on top of climatology's dashed line for most of the period shown. The overfit tree is the one model that visibly does something different — jagged, noise-chasing behavior, not distinguishable skill. The VAE carves out its own identity too, but as instability: large, sudden excursions (e.g. the drop toward 300 in mid-2025) that do not correspond to anything in the actual price path.*

![Figure 2](predictor_v1/65_architecture_bakeoff_EURUSDX.png)
*Figure 2. EURUSD=X. The same pattern: climatology, the RL forecaster, and the GAN move as a single cluster through nearly the entire recent-period panel. The tree remains the visibly distinct, noisy outlier; the VAE again shows occasional sharp departures from the rest of the field.*

![Figure 3](predictor_v1/65_architecture_bakeoff_XLF.png)
*Figure 3. XLF. Same result a third time: the RL forecaster and GAN sit on climatology throughout, the tree is noisy, and the VAE's excursions are the most visible during the 2008–2009 window in the full-history panel — a period of genuine, unusual instability in the underlying instrument, which the VAE's own forecasts echo rather than smooth over.*

In no instrument, at any point in either panel, does any architecture visibly and durably separate itself from climatology in the direction of better tracking actual price. Two of the four non-trivial architectures essentially rediscover climatology under this fair training budget; the third (the tree) does something different, but that something is noise, not skill; the fourth (the VAE) does something different too, but that something looks more like a liability than an edge.

## 5. Experiment 2: Sweeping Training-Window Size Past the Predictability Limit

### 5.1 Design

Experiment 2 holds the segmentation of test days completely fixed — the same test points are predicted at every step of the sweep — and varies only how much training history immediately precedes each of them, as a multiple of the instrument's own τ\*:

<div class="eqn-row"><span class="eqn-spacer"></span><img src="predictor_v1/eq14_figs/eq4_window_sweep.svg" alt="Equation 6"><span class="eqn-num">(6)</span></div>

The first test segment for every multiple m is anchored to the same date across the entire sweep (fixed by the largest training window in the sweep, m = 8), so that every panel a reader compares is predicting the identical stretch of actual history — the only thing that changes between panels is how much, and how stale, the training data behind each model was.

### 5.2 The historical parallel this experiment is modeled on

Lorenz's predictability-limit result was, at the time it was derived, a purely theoretical statement about error growth in a chaotic system — no numerical weather model was required to state it. It was only afterward, once actual numerical weather prediction models were built and run operationally, that the same limit was confirmed empirically: forecast skill did not degrade smoothly and indefinitely as lead time grew, nor did it hold up cleanly forever given a good enough model — it held up reasonably well and then, in the region the theory had already identified, began to fail, regardless of which specific model was doing the forecasting. This paper's second experiment is built to reproduce that exact structure for the financial predictability limit measured in Paper 11: if that limit is genuinely structural rather than a property specific to any one estimation method, then the same five very different architectures from Experiment 1, trained on data reaching further and further past it, should all begin failing in roughly the same place — not because of anything specific to their own designs, but because of what happens to the data itself once it crosses the boundary.

### 5.3 Instrument coverage

Because this experiment does not depend on a data-hungry architecture having enough rows within a tight window — it deliberately tests what happens as that window grows — it is run across all 12 of Paper 13's instruments. Table 1 lists each instrument's predictability limit and the training-window sizes swept.

**Table 1.** Predictability limit and swept training-window sizes (days), selected multiples shown; the full sweep uses eight multiples (0.5, 1, 1.5, 2, 3, 4, 6, 8) × τ\* for every instrument.

| Ticker | Horizon (days) | τ\* (days) | 0.5x | 1x | 2x | 4x | 8x |
|---|---|---|---|---|---|---|---|
| GLD | 189 | 22 | 11 | 22 | 44 | 88 | 176 |
| JPM | 252 | 23 | 12 | 23 | 46 | 92 | 184 |
| AAPL | 252 | 22 | 11 | 22 | 44 | 88 | 176 |
| XLK | 189 | 22 | 11 | 22 | 44 | 88 | 176 |
| EURUSD=X | 189 | 43 | 22 | 43 | 86 | 172 | 344 |
| IWM | 21 | 23 | 12 | 23 | 46 | 92 | 184 |
| MSFT | 189 | 63 | 32 | 63 | 126 | 252 | 504 |
| QQQ | 21 | 22 | 11 | 22 | 44 | 88 | 176 |
| SPY | 189 | 22 | 11 | 22 | 44 | 88 | 176 |
| XLE | 252 | 27 | 14 | 27 | 54 | 108 | 216 |
| XLF | 189 | 33 | 16 | 33 | 66 | 132 | 264 |
| XOM | 63 | 27 | 14 | 27 | 54 | 108 | 216 |

## 6. Experiment 2 Results

The full sweep for every instrument uses eight stacked panels, one per swept multiple of τ\*, all predicting the identical recent-two-year test stretch, all five models overlaid against actual price — these complete figures are generated by `66_window_sweep_bakeoff.py` for all twelve instruments and are available in full in the repository. For this paper, a compact three-panel highlight (0.5x, 2x, and 8x τ\* — within the limit, just past it, and deep beyond it — laid out side by side) is shown inline below for five representative instruments, generated by the companion script `67_window_sweep_highlights.py` using identical models, data, and windowing logic.

![Figure 4](predictor_v1/67_window_sweep_highlight_MSFT.png)
*Figure 4. MSFT. At 0.5x τ\*, the climatology/RL/GAN/VAE cluster tracks real turning points reasonably well — visible co-movement with actual price through the 2025-04 to 2025-07 dip and the early-2026 dip. By 8x, that same cluster has smoothed into its own long-run trend that increasingly ignores those same dips entirely, riding a slower path decoupled from the real zigzags in the black curve. The overfit tree, notably, does not show this pattern at any window size — it stays noisy throughout, because overfitting to whatever is in front of it has no "collapse toward the mean" failure mode the way a smoothing model does.*

![Figure 5](predictor_v1/67_window_sweep_highlight_EURUSDX.png)
*Figure 5. EURUSD=X. The clearest case in the panel: at 0.5x, the model cluster tracks the real exchange-rate path closely. By 2x a persistent overshoot has appeared, and by 8x the model cluster sits at roughly 1.20–1.25 for months while the real rate sits at 1.15–1.20 — a gap that widens rather than merely growing noisier as the window grows.*

![Figure 6](predictor_v1/67_window_sweep_highlight_JPM.png)
*Figure 6. JPM. The same gradual detachment, building across the three panels rather than appearing discontinuously — consistent with the common-sense expectation that a real predictability limit should produce a build, not an abrupt flatline, at the boundary.*

![Figure 7](predictor_v1/67_window_sweep_highlight_XLF.png)
*Figure 7. XLF. A visible, growing upward bias appears in the model cluster by 8x, most apparent in the 2025-10 through 2026-04 stretch, while the tree remains its own noisy, non-participating self throughout.*

![Figure 8](predictor_v1/67_window_sweep_highlight_QQQ.png)
*Figure 8. QQQ, one of the two 21-day-horizon instruments Paper 13 already flagged as inherently noisier. The drift effect is present but far more muted here than in the longer-horizon instruments above — consistent with Paper 13's own Section 4.2 finding that a shorter horizon leaves less structure in the target curve for any comparison, fresh-versus-stale or otherwise, to be legible against.*

Across all twelve instruments, the pattern is consistent: the longer-horizon instruments (GLD, JPM, AAPL, XLK, EURUSD=X, MSFT, SPY, XLE, XLF, XOM) show a visible, progressive detachment that builds panel over panel and becomes unmistakable by four to eight times τ\*; the two 21-day-horizon instruments (IWM, QQQ) show the same direction of effect at reduced, harder-to-read magnitude, for the same reason Paper 13 already gave for their noisier fresh-versus-stale comparison. In every instrument, the effect is a gradual build rather than a discontinuous cliff — precisely what a genuine, structural predictability limit should look like, and precisely what Lorenz's own theoretical limit looked like once real forecasting systems began running into it.

## 7. Discussion

Read together, these two experiments answer the question this paper opened with in a way neither could alone. Experiment 1 shows that within a fair, non-stale training budget, architectural sophistication does not buy separation from the simplest possible model — climatology is not beaten, and two of the four non-trivial architectures effectively become climatology under this constraint. Experiment 2 shows that the reason is not incidental to these particular five models: push the same architectures' training data past the boundary Paper 11 already measured by a method with no model in it at all, and all of them — regardless of internal mechanism — begin drifting away from reality, gradually, and the drift's onset lines up with the same τ\*.

This is the same two-step validation this paper's introduction described for Lorenz's own predictability limit: a structural ceiling, first identified without reference to any specific model, later confirmed not by further argument but by watching real, working models run into it regardless of their design. This paper's contribution is to have run that second step for the financial predictability limit specifically, using architectures chosen to be as different from one another as reasonably possible — a lookup-table average, a memorizing tree, a reward-driven policy, an adversarial pair, and a latent-variable generative model — precisely so that a shared failure point could not be dismissed as an artifact of any one of them.

**A genuine, non-obvious asymmetry is worth stating plainly rather than smoothing over.** The climatology-collapse pattern in Experiment 2 is specific to architectures that have some form of averaging or smoothing built into how they learn — climatology by definition, and the RL forecaster, GAN, and VAE by virtue of being continuous, smooth function approximators trained to minimize an expected error. The overfit tree, which has no such inductive bias and instead memorizes whatever is directly in front of it, does not show this pattern at any window size tested, in any instrument. This means the mechanism this paper demonstrates — sophistication quietly reducing to climatology once training data spans multiple regimes — is a property of a specific and common class of models, not of "AI/ML" as an undifferentiated category. A practitioner using a memorizing, non-smoothing model would not be protected by this paper's own diagnosis; they would simply be overfitting badly at every window size, which is Paper 13's finding, not this one.

## 8. Limitations

- **Scale of the RL/GAN/VAE implementations.** All three are linear-parameter, numpy-native implementations, appropriately sized to training windows of 11 to 504 rows, not deep networks. This paper's finding characterizes these architectural families at the scale the predictability limit itself permits; it does not claim the same result would hold for a large-scale, heavily-parameterized version of the same architecture trained on genuinely independent data from many separate instruments or markets.
- **A gradual build, not a sharp cliff.** Every instrument in Experiment 2 shows a progressive, not discontinuous, degradation. This is the expected and, on reflection, the only sensible shape for a genuine predictability limit to take — but it also means "beyond the limit" is a matter of degree in this paper's own results, not a bright line a reader can point to on any single panel.
- **Short-horizon noisiness carried over from Paper 13.** IWM and QQQ's 21-day horizon makes both experiments' comparisons harder to read by eye for exactly these two instruments, for the same reason Paper 13 already documented.
- **Instrument coverage differs by experiment.** Experiment 1 is deliberately restricted to the three instruments with the largest measured predictability limits, for data-sufficiency reasons stated in Section 3.1; Experiment 2 covers all 12.
- **No metrics, by design, not by omission.** As in Paper 13, this paper reports no significance test or skill score in either experiment; the standing position of this research program is that a question this practical is more honestly settled by direct visual demonstration.

## 9. Conclusion

The claim this paper set out to test — that a sufficiently sophisticated, sufficiently data-fed AI/ML architecture might simply predict past a measured predictability limit — does not hold up under either of the two tests this paper ran. Given a fair, non-stale training budget sized to respect the limit, no architecture among five genuinely different families beats the simplest possible model, and two of them quietly become it. Given training data that grows to span multiples of that same limit, all five degrade, gradually but unmistakably, starting in roughly the same place — regardless of whether the underlying mechanism is a lookup-table average, an adversarial pair, or a latent-variable generative model. Lorenz's own predictability limit was a theoretical result long before any model could test it, and it took real forecasting systems running into that same wall to turn it from theory into an operational fact of meteorological practice. This paper does the equivalent for the financial predictability limit this research program has already measured: not an argument that sophistication cannot buy its way past a structural ceiling, but a demonstration, architecture by architecture, that it does not.

---

## References

Lorenz, E. N. (1963). Deterministic Nonperiodic Flow. *Journal of the Atmospheric Sciences*, 20(2), 130–141.

Lorenz, E. N. (1969). The predictability of a flow which possesses many scales of motion. *Tellus*, 21(3), 289–307.

Williams, R. J. (1992). Simple statistical gradient-following algorithms for connectionist reinforcement learning. *Machine Learning*, 8, 229–256.

Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D., Ozair, S., Courville, A., & Bengio, Y. (2014). Generative Adversarial Networks. *Advances in Neural Information Processing Systems (NeurIPS)*.

Kingma, D. P., & Welling, M. (2014). Auto-Encoding Variational Bayes. *International Conference on Learning Representations (ICLR)*.

Breiman, L., Friedman, J. H., Olshen, R. A., & Stone, C. J. (1984). *Classification and Regression Trees*. Wadsworth.

Ramanathan, A. (2026). Empirical Predictability Limits of Financial Markets via Correlated–Decorrelated Structure Function Decomposition: A Departure from Atmospheric Turbulence Theory. *Zenodo* (Paper 11, this series). https://doi.org/10.5281/zenodo.21373459

Ramanathan, A. (2026). A Master-Model Framework for Regime-Conditioned Price Forecasting: Real Statistical Skill, and Why It Mostly Isn't Alpha. *Zenodo* (Paper 12, this series). https://doi.org/10.5281/zenodo.21454884

Ramanathan, A. (2026). Predictability Limits as Regime Detectors: A Practical Rule for How Much History an AI/ML Model Should Train On. *Zenodo* (Paper 13, this series). https://doi.org/10.5281/zenodo.21482869

---

## Code and Data Availability

Experiment 1's figures are generated entirely by `notebooks/predictor_v1/65_architecture_bakeoff.py`; Experiment 2's complete, twelve-instrument, eight-panel-per-instrument sweep by `notebooks/predictor_v1/66_window_sweep_bakeoff.py`, with this paper's compact five-instrument, three-panel highlight figures generated separately by `notebooks/predictor_v1/67_window_sweep_highlights.py` (identical models and windowing logic, a different figure layout chosen to fit a printed page); this paper's equations by `notebooks/predictor_v1/render_paper14_equations.py`. All four are self-contained and available at `notebooks/predictor_v1/` in the `quantarram/quant-regime-research` repository. Both experiments build directly on Paper 11's predictability-limit results (`predictability_paper/results_correlated_decorrelated.json`) and reuse Paper 13's half-window design and overfit-tree diagnostic; neither script requires a deep learning framework — all three non-trivial architectures (the RL forecaster, the conditional GAN, and the conditional VAE) are implemented from scratch in numpy, with manual gradients, at a scale appropriate to the training-window sizes the predictability limit itself dictates.
