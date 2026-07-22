# How Much History Should Your Model Actually See? I Found the Answer Sitting in an Old Paper of Mine.

Every team that fits a machine learning model to real-world data eventually has to answer a question they usually never write down anywhere: how much history should this thing actually train on? A year? Five years? Whatever happened to fit in the last training run? In two very different fields now, I've watched smart people answer this by habit, and almost never by anything specific to the system they're actually modeling.

I spent years of my PhD measuring something that sounds abstract until you need it: how far ahead atmospheric turbulence stays genuinely predictable before it dissolves into noise. Years later, in an unrelated stretch of quant research, I built the same kind of measurement for financial markets — not by assuming a theory, but by directly, empirically clocking how many days ahead an instrument's own price dynamics stay connected to themselves before that connection fades. I called it a predictability limit, reported it, and moved on. It sat in a paper as a nice fact about market structure.

Then I did something with it I hadn't planned on.

## The experiment: deliberately break a model on purpose

Here's the design, in one picture:

![Fresh vs. stale training design](predictor_v1/schematic_fresh_vs_stale_design.png)

Take that predictability-limit number — call it τ* — and use half of it as a training window. Fit a model on the most recent half-window of data ("fresh"), and separately fit the identical kind of model on an equally-sized window pulled from more than *twice* that limit in the past ("stale"). Point both models at the same test window right after the fresh one, and just look at what they each predict.

The catch, and the fun part: the model I used is deliberately terrible in one specific way. It's a decision tree with zero regularization — no depth limit, no minimum leaf size — which means it doesn't approximately fit its training window, it *memorizes it exactly*. I checked: training error of precisely 0.000000, one leaf per training day, every time. Normally that's the textbook mistake — a model that's learned nothing except its own training data's noise.

But that's exactly what makes it a good detector here. A well-behaved, regularized model is built to generalize, which means it can quietly paper over a real regime change by just being mediocre everywhere. A model that has memorized every idiosyncrasy of one specific window has no such safety net. If the next window is genuinely the same regime, those idiosyncrasies still carry real information forward. If the regime has shifted, they shouldn't — and they don't.

## What it looks like when you just watch the curves

I didn't run a single significance test for this paper. No p-values, no null hypotheses, no resampling. Just the model's predicted curve laid directly against what actually happened, for every one of the 12 instruments I could test this on, both for the full feature set and for a version using nothing but the instrument's own past returns.

The fresh-trained line tracks reality closely — including through sharp reversals — in every single instrument. The stale-trained line, given data from the same amount of history but from the wrong point in time, visibly comes apart. AAPL is the cleanest example: trained on data from more than double its own predictability limit ago, the model's predictions overshoot the real move by close to double in multiple stretches, while the fresh-trained version tracks it closely the whole way through.

That's the entire finding, and I think it's more useful *because* it's small: measure how long your own data stays self-similar, train on about half of that, and stop trusting anything trained from meaningfully further back.

## Why I think this is bigger than finance

Nothing in that mechanism refers to prices, returns, or markets. It's a concrete answer to the training-window and concept-drift problem that sits underneath any AI/ML pipeline built on data that doesn't stay still — weather models, industrial sensors, recommender systems watching user behavior shift, epidemiological forecasts, anything where a team currently picks a lookback window because it's convention, not because they measured the system's own memory.

This idea already crossed one field boundary once, quietly, from atmospheric turbulence theory into financial markets, in my own research history. I don't see a reason it has to stop there.

Full preprint: https://zenodo.org/records/21482869
Code and all 12 instruments' curves: https://github.com/quantarram/quant-regime-research

*Independent quantitative research. Not investment advice.*
