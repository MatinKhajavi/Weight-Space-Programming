Brake-target control shows a specificity effect: +0.4 against A and -24.6 against C, both with tight error bars. C is restored and verified. The GSM miss is a wrong answer (72), not a formatting problem, since there are no format near-misses. Induction mass under C is strong across the loop windows, ranging from about 11 to 20.

Induction mass under C holds up across the loop windows, though the raw attention values come out lower than my earlier notes suggested, so I'll treat those earlier figures as a different definition. Window 19.0 has almost none, so it doesn't count as a copier. With C settled and the controls strong, I'm weighing whether the remaining 25 passes and 32 Python runs are better spent on the GSM errors, which look like genuine reasoning slips rather than formatting problems.

I'm comparing my recollection of Qwen3-0.6B's reported scores against our numbers: GSM8K around 66-73% looks reasonable for this size, but MATH-500 at 53% is well below the roughly 77.6% I remember for thinking mode. That gap suggests something specific is still going wrong on the math side, possibly in how the eval is configured, such as the presence penalty of 1.5.

I'm comparing the recommended sampling settings for thinking mode, including the presence penalty, against what a healthy 0.6B model should score on MATH-500, around 77%. Our edited 0.8B sits at 53%, and the original at about 8% from looping, which makes me wonder whether the model was deliberately damaged to produce a loop pathology.

I'm wondering whether the model was deliberately damaged to produce loops, in which case my repair may only have undone part of the damage and other components could still be off. Qwen3.5 small models might also simply be loop-prone in thinking mode, so I'm considering that as the more likely explanation.

I'm checking whether the model was deliberately damaged by looking at the weight statistics: the down-projection column for neuron 22.573 against other layer-22 neurons, and the o_proj norms for heads 15.7, 23.0, and 23.5 against the rest, since amplified or halved components should stand out.

I'm going to check the weights for anomalies by computing per-head output-projection norms, per-neuron MLP down-projection norms, and the q/k norm scales, then flagging outliers within each layer. That would show whether 22.573 and heads 15.7, 23.0, and 23.5 stand out from their neighbors, which could mean other components were altered too.

I'm noting that the linear-attention layers also carry per-head parameters like A_log and dt_bias, where a subtle change could alter memory length, but I can't easily judge normality there without a reference. The next step is an anomaly scan: computing per-layer distributions of per-head norms and flagging heads beyond |z| > 3, then doing the same for the per-neuron MLP down-projection norms.

Checking the weights for a deliberately damaged component is cheap, one Python run. If the creators left the model alone, the scan will only turn up natural outliers, but it could also expose a component that explains the remaining math failures, so I'll compare neuron 22.573's column norm and heads 15.7, 23.0, and 23.5 against their layer peers, then run a broader outlier pass.

Now I'm checking the weights for signs of deliberate damage, starting with the per-head and per-neuron norms in the attention and MLP layers.

Checking the weight layout for the attention projections next: I'm working out the per-head sizes for q, k, v, and o, and how the query and gate outputs are arranged. That tells me how to pull out each head's output-projection block so I can compare head 15.7, 23.0, and 23.5 against the rest of their layers.

Next I'm checking the per-head and per-neuron norms, starting with where neuron 22.573 ranks among layer 22's down-projection columns and how its up/gate rows compare.

