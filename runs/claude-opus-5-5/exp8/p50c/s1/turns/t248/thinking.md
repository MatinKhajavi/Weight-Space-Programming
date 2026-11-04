I'm comparing the scores within each source and within each MATH level. The my_thought style comes out ahead of a_thinking in both GSM and MATH, and the gap holds across the levels too.

I'm checking whether the gap holds inside each MATH level, and it does, though the L5 sample is tiny. The my_thought style shows up far more at the lower levels, so the style choice seems tied to difficulty, which makes within-level confounding a real concern. Testing that would take several samples per problem with different seeds.

I'm checking whether the style gap survives once I test it properly: the my_thought texts might be shorter and less nested, which would point to a more efficient reasoning style. To separate style from problem identity, I'd generate several samples per problem and compare accuracy within each problem across the two openings, since with a roughly 20% rate of my_thought openings, most problems would show both styles across four samples.

I'm weighing two ways to steer the model's opening choice. One is a narrow edit that pushes the logit toward " my" over " a" right after "Here's" at the start of the thinking, which would mean wiring a rank-1 change into an MLP that fires only at that position. The other is to find a mid-layer "mode" direction by comparing average residual activations between the my_thought and a_thinking texts.

I'm working through whether a steering vector could be applied across the whole generation, or whether it would need to be tied to a single position. I'm weighing whether any component is constantly active that could carry a fixed injection, and I'm not sure that exists in a model without biases, so I'm going back to the idea of finding the internal mode direction instead.

I'm weighing whether the teacher-generated "thinking process" style could explain the opening choice, with "my thought process" coming from a different data source. The mode seems set by the first tokens, so shifting it would mean raising the probability of " my" after "Here's" from about 0.2. I'm checking whether that falls within the scope of what I'm meant to do, and it doesn't: this isn't about making the model skip thinking or cut it short, and it isn't a global change to the output distribution either.

I'm weighing whether steering the reasoning mode counts as a targeted fix or just a surface hack. If the failures cluster in the "a thinking process" mode and I can trace that choice to a specific direction or set of neurons with evidence, a mechanistic edit could be justified, though a skeptic would still call it superficial.

I'm checking whether the mode actually causes the difference, which means comparing success within the same problem across the two openings. Then I want to see whether the original model's openings line up with the same seeds, so I can compare per-style accuracy directly.

I'm planning a cheap causal check: sample four generations on 128 fresh problems and compare accuracy within each problem across the two openings. I'd also look at how often each opening shows up per problem. Editing the opening token directly is an option I'm still weighing.

I'm going to compare the my_thought and a_thinking texts on length, nesting depth, and how often they say "Wait" or "Maybe", starting with a short excerpt from each in the GSM set to see the difference directly.

