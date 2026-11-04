I'm checking whether the cross-tab numbers show the same pattern across the ablations: the ok-to-wrong shifts land around 30 to 36 in each comparison. I'm still deciding whether that counts as damage.

I'm weighing whether the ok-to-wrong flips are just churn. The null comparison (gen_21 against C) gives roughly 35 flips each way per 512 answers, so X1's 33 looks like ordinary churn rather than a specific effect. The net shift is still worth checking, since X1 has more wrong answers overall.

I'm working out whether the X1 imbalance means anything. Under independent re-draws the ok↔wrong flows would balance, but X1 has 33 ok→wrong against 18 wrong→ok, and cut→ok runs higher than ok→cut. So it looks like X1 moves probability from cut toward wrong. Whether that's damage or just a shift is still open. Next I'm checking the same flows for X4.

Checking the X4 counts: wrong goes from 75 to 73, and cut rises from 126 to 146. Neither ablation changes the picture, so every head examined seems to carry a real function. That makes me think the loop problem is capability-limited rather than something one head controls.

I'm now thinking about whether the brake neuron's doubled suppression could interfere with ending the thinking. It fires at new-thought line starts and pushes down words like Wait and Hmm, which seems fine, but it also nudges down </think> slightly at those same points. Under the current setup correct answers still terminate cleanly, though some could be affected.

I'm now asking whether the brake neuron's suppression of the end-of-thinking token could cause problems, since it fires at the same line starts where the thinking would normally close. The earlier check suggests the effect is small in the loop windows, but I'm wondering whether the closing point itself, where the final answer is announced, might be the place where a commit failure starts.

I'm looking at whether the commit failures, around 25 of the 126 cuts, are worth fixing. If the model could finish those answers with the committed result, I'd expect a gain of about three points. The useful question is what follows a commit statement in the failing texts, like a "Wait" or a "Maybe I should double check", and what drives that continuation.

I'm going back over the cut texts under C in H2, checking whether they're exact loops and what the repeated units look like, since the earlier split was roughly 47 exact loops out of 74.

I'm wondering whether the loop problem could be tackled by finding a repetition-tracking neuron, one whose activation grows with each repeated cycle. If the forward-pass activations for every layer were saved on the C loop windows, I could search for neurons that climb across cycles.

I'm wondering whether the brake neuron might be suppressing the model's own loop-breaking attempts, since the "Wait" it pushes down could be a way the model tries to escape repetition. Checking the saved forward-pass data in fwd_18 should show whether any neuron's activation grows with each repeated cycle.

I'm planning to find neurons across all layers whose activation rises with each repeated loop cycle, comparing matched tokens in consecutive cycles so only the repetition count changes. The idea is that neurons consistently climbing with repetition could be the model's own loop-sensing signal. I'd then check whether their output directions push toward the Wait-family tokens that the brake is suppressing.

I'm planning to look across the layers for neurons whose activation grows with each loop cycle, comparing the same token at the same offset so that only the repetition count changes. If some of those neurons push toward break tokens like "So", "Therefore", or "</think>", they would be the model's own loop brakes, and amplifying them could be a targeted fix. The loss is already near-deterministic from the first copy, so I'm also wondering whether the attention heads show growing contributions as the cycles repeat.

Next I'm checking whether the repeated units tell me anything: if fwd_18's windows hold enough cycles, I could look for neurons whose activation climbs with each repetition. The 6 GB memory limit is the constraint, so I need to estimate how many tokens I can afford to save per layer.

Checking the stored activations is the next step: the existing forward passes already hold MLP activations for every layer on the loop windows of the original and A-window runs, so I can look at those before spending memory on new captures. The budget is tight, though, at roughly 17k tokens if I save everything, so I'd restrict the capture to a subset of layers.

I'm checking the stored activations from fwd_18 next, with the aim of finding neurons whose activation climbs across repeated loop cycles at matched offsets. Once I have them, I'll project their output directions onto the vocabulary to see whether any of them push toward break tokens.

I'm planning to check how each candidate neuron's change pushes the copy-target logit, to see whether it brakes or reinforces the repetition. For the vocabulary projection I'll write a simple helper using the tied embedding matrix and the final norm weight, skipping the RMS term since I only need relative rankings. The embedding table is about 1 GB in float32, which fits the budget.

I'm going to check what fwd_18 contains first, its meta and array shapes, along with the window summary file, and print only the structure rather than the long texts.

