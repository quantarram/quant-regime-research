# Can a Smarter Model Beat a Real Predictability Limit? I Ran the Experiment. It Can't.

There's a piece of AI/ML conventional wisdom I've never bought, and I finally built the experiment to test it properly.

The assumption, mostly unstated but everywhere: given enough data and a powerful enough model, you can eventually predict almost anything. It's a modern, quieter version of an old idea — that a sufficiently complete description of a system, fed into a sufficiently powerful machine, renders its future fully knowable. Chaos theory already settled that question in the physical sciences, decades ago, and it settled it the hard way: not by argument, but by real weather models, built to be dramatically more sophisticated than the ones before them, running into the exact same forecasting wall a much simpler theory had already predicted years earlier. Some systems have a genuine, structural ceiling on how far ahead they can be known. No amount of scale fixes that, because the limit isn't a property of the model. It's a property of the dynamics.

My last paper measured a ceiling exactly like that for financial markets — directly and empirically, no model involved in the measurement itself: how many days ahead an instrument's own price behavior stays genuinely connected to itself before that connection fades into noise. So this paper asked the obvious next question, the one that actually matters if you take the first result seriously: can real architectural sophistication buy its way past that measured ceiling anyway?

## Test one: a fair fight, no cheating on data

Five architecturally different models, given the exact same training budget — sized to stay safely inside the measured limit, identical for every model, so none of them is quietly seeing more or staler data than the others: a plain average, a decision tree with zero regularization, a reinforcement-learning forecaster trained by policy gradient, a conditional generative adversarial network, and a conditional variational autoencoder. Same instruments, same walk-forward evaluation, same fair shot.

The plain average won. Two of the four more sophisticated models didn't just lose to it — they converged onto it almost exactly, as if they'd rediscovered it from scratch. The decision tree did something different, but that something was noise, not skill. The generative model did something different too, and that something was instability, not an edge.

## Test two: closing the obvious objection

The obvious pushback to test one: maybe nothing beat the average because those three fancier models were kept deliberately simple — linear, no hidden layers, nothing a machine learning engineer would call "deep." So I built genuinely nonlinear versions of all three, by hand, with a real hidden layer and manually derived backpropagation, validated first on synthetic data to make sure the math was actually correct before trusting it on anything real. Same tiny training budget, exactly. No extra data given to the deeper models — feeding them more data to make them properly trainable would have meant training past the very limit under test, which defeats the point entirely.

Depth changed nothing. The reinforcement-learning forecaster and the adversarial network still converged onto the plain average, hidden layer or not. The generative model was still unstable, hidden layer or not — in one instrument its deeper version was measurably *more* erratic than its simpler one, not less.

## Test three: what happens when you keep feeding them data anyway

This is the one I think is most convincing, because it directly answers a fair objection to the first two tests: maybe the training window was just too small for *any* model to show real skill, independent of any genuine limit — and if that were true, giving the same models progressively more data should eventually let real skill emerge, the way an ordinary statistical model improves with sample size.

It doesn't. I took the same five models and swept their training-window size from well inside the measured limit out to eight times beyond it, holding the exact days being predicted fixed throughout so the only thing that changed was how much, and how stale, the training data behind each model was. Error didn't fall as more data arrived. It climbed — roughly doubling in most instruments — the opposite of what more data is supposed to do.

![Architecture bake-off, before and after the limit](predictor_v1/67_window_sweep_highlight_EURUSDX.png)

You can see it directly above, no statistics needed: at half the predictability limit, every model tracks the real exchange rate closely. By twice the limit, a gap has opened. By eight times the limit, the models are stuck near 1.20–1.25 for months while the real rate sits at 1.15–1.20 — a gap that keeps widening, not one that just gets noisier.

## Why I think this is bigger than finance

The atmospheric predictability limit this whole idea traces back to was never settled by theory alone. An empirical estimate came first, from real (if primitive) weather models in the 1960s. A formal theoretical account of why such a limit should exist followed a few years later. And then, as weather models grew dramatically more sophisticated over the following decades — finer resolution, more data, far more compute — the same ceiling held anyway, confirmed not by further argument but by watching real, increasingly capable systems run straight into it.

This paper is the same shape of evidence, for a different field: architecture, depth, and raw training data were all tested against a measured ceiling, and none of them bought a way past it. Sophistication can get you *to* the limit. It can't get you *through* it. That's not a disappointing result. It's the whole point.

Full preprint: https://zenodo.org/records/21696948
Code, all instruments, every figure: https://github.com/quantarram/quant-regime-research

*Independent quantitative research. Not investment advice.*
