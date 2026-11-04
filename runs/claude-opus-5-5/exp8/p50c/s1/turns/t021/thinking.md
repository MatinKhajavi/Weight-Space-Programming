I'm seeing the outputs collapse into short lines repeated word for word, often with "Wait" in them, so the model looks stuck in a loop of checking itself. That starts at around 13 to 19k characters. I suspect this is a common looping failure in small models, and I'm weighing a few possible causes, like induction-style copying in the full attention layers or a saturated state in the linear attention, plus a bias toward "Wait".

I'm weighing which components drive the verbatim copying in these loops, starting with the full-attention layers that might be running induction-style copying. I'm also considering whether the "Wait" token is the main repair target without tipping into forcing thinking to close early.

I'm weighing whether the repetition is driven by induction-style copying in the full-attention layers or by the delta-rule state in the linear layers, since both could implement the verbatim recall seen in these loops. Checking attention patterns would mean feeding in a text that already reaches the loop onset, and I'm considering whether to reuse a real transcript for that or to work from the weights directly.

I'm checking whether the presence penalty is making the loops worse, since at 1.5 it pushes down any token already seen, which in long reasoning is nearly every common word.

I'm weighing whether to fix the copying itself, meaning the output that tends to repeat a previous line's continuation, or to nudge the model toward a stop signal after a repeated pattern. The sampling settings are fairly sharp, so once a loop has started each token is predicted with high confidence and escaping it looks hard.

I'm reading about "repetition neurons" in MLP layers, whose activation grows as a repetition continues and whose deactivation reduces it. I'm also considering induction heads, which drive the repetition but are needed for copying numbers in math, so dampening them has a cost.

I'm now planning a localization approach: run a teacher-forced pass over a looping transcript and collect per-token MLP activations, hidden states, and full-attention maps, then compare tokens inside the repeated section against the first occurrence to find neurons that fire more strongly during repetition. I'd then test those candidates causally by editing their weights and checking whether the loop probability drops.

I'm planning to check which repetition-linked MLP neurons drive the loop, comparing activations inside the repeated section against the first occurrence and then testing candidates by editing their weights. The loop onset sits around 5k tokens, and I'm considering a shorter excerpt with the chat template, since a self-sustaining loop should show the mechanism without the full prefix.

Now I'm weighing how to build a shorter test text: a problem prompt plus about 300 tokens of pre-loop reasoning and several loop repetitions, which should still show the copying mechanism. Chat-template handling of thinking is a side question; raw text with my own formatting is simpler. The main constraint is output length, since a roughly 2000-character excerpt is manageable to type, so I'll plan two such texts with attention maps and perhaps a few more without.

I'm weighing whether the averaged activations can separate looping samples from finished ones, since looping ones carry loop content in their first 8192 tokens when the loop starts early. The idea is to regress each sample's average on the fraction of those tokens that sit inside the loop, using the per-sample onset, and compare within the same cut class.

I'm checking whether averaged activations split looping from non-looping samples in a within-cut comparison, which should weaken the difficulty confound a bit. Next I want to look at the saved activation files.

