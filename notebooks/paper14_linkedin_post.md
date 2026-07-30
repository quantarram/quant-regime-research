There's a piece of AI/ML conventional wisdom I've never bought, and I finally built the experiment to test it properly.

The assumption, mostly unstated but everywhere: given enough data and a powerful enough model, you can eventually predict almost anything. Chaos theory already settled that question in the physical sciences, decades ago — and it settled it the hard way, not by argument, but by real weather models, built to be dramatically more sophisticated than the ones before them, running into the exact same forecasting wall a much simpler theory had already predicted years earlier. Some systems have a real, structural ceiling on how far ahead they can be known. No amount of scale fixes that, because the limit isn't in the model. It's in the dynamics.

My last paper measured that kind of ceiling directly for financial markets — how many days ahead an instrument's own price behavior stays genuinely connected to itself before the connection fades into noise. So this paper asked the obvious next question: can real architectural sophistication buy its way past that measured ceiling anyway?

Five architecturally different models — a plain average, a decision tree with zero restraint, a reinforcement-learning forecaster, an adversarial network, a generative model — all given the exact same fair, tiny training budget the ceiling allows. Then, because "these are just simple implementations" is the obvious objection, I hand-built a real hidden layer of nonlinearity into three of them and reran the whole comparison. Then I took the same five models and kept feeding them more and more data, the way a team chasing "more data" always does, and watched what happened as they crossed the same limit.

![Architecture bake-off, before and after the limit](predictor_v1/67_window_sweep_highlight_EURUSDX.png)

The plain average won, every time. Depth didn't change that. And feeding the fancier models more data didn't help them catch up — it made them measurably worse, error climbing instead of falling, the opposite of what more data is supposed to do.

The ceiling held, exactly the way I expected it to. Not because the models were bad. Because a structural limit doesn't care how good the model is.

Preprint: https://zenodo.org/records/21696948
Code, all instruments, every figure: https://github.com/quantarram/quant-regime-research

Independent quantitative research. Not investment advice.
