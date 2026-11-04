I'm listing each head's direct effect on the next-token logit, split into in-loop and pre-loop values. The main suppressors are heads 19.4, 15.3, 23.3, 19.6, and 15.4, and since their in-loop and pre-loop values are similar, they look like general calibration heads that push down the actual next token in both settings.

Now I'm weighing the promoters: 23.4 stands out as the largest loop-specific boost, while 23.6 and 23.2 stay strong in both settings. The suppressors are small and non-specific, so scaling them up would likely just flatten normal prediction, which doesn't look worthwhile. Ablating 23.4 would also cost normal text, so the head-level route seems largely played out.

I'm now looking at a different angle for the remaining loops: the loop entry is probabilistic, so the aim is to lower the per-line chance of repeating the previous line. I also want the model to finish reasoning sooner, since most correct answers are short but a tail runs past 18k tokens. The "Maybe" spirals, interpretation doubts that stack up line by line, are another thing I'm checking, since their rate is much higher in the cut texts.

I'm now trying to pin down where the doubt impulse comes from, since the "Wait" brake only blocks the word, not the urge behind it. The idea is to find the residual-stream direction at new-thought line starts that feeds Wait, Maybe, and similar words, and to check whether neuron 22.573, whose activation dips most at those starts, reads that same direction through its input weights.

I'm thinking about whether a residual-stream direction at layers 19–22 predicts the doubt markers (Wait, Maybe, Hmm, Actually, But) versus neutral openers like So or Let, which could be computed from teacher-forced hidden states at line starts. The catch is that my existing windows only give around 133 doubt-relevant line starts, so I'd need a larger teacher-forced pass to get a usable signal, and that would be long.

I'm weighing whether a larger teacher-forced pass is worth the output cost, since it could be tens of thousands of characters. I'm also considering a cheaper route: have the generation run save residual-stream and MLP activations per answer, then look for neurons whose averages track looping or cutoff. That would be correlational and could be confounded by problem difficulty.

I'm looking for neurons in layers 16–23 whose output weights point toward "Maybe" and whose activations could act as either a brake or a driver of that doubt impulse, checking their sign pattern across the existing teacher-forced windows.

I'm going to search the neurons in layers 16 to 23 by how strongly their down-projection columns push the " Maybe" and "Maybe" tokens, then check how those neurons activate at line starts in the cut windows. The V2 change seems to have moved some line starts from "Maybe" to "Okay," so I want to know whether the doubt impulse is being redirected rather than removed.

I'm weighing a set of options to see which might give a bigger gain than the current edit, which is already strong on held-out data. One candidate is scaling up the brake neurons, which is cheap to test in a single pass, though I'll need to check it doesn't disturb normal text.

I'm weighing a partial ablation of 23.4 at half strength, since full removal hurt when combined with the brake. The idea is that the strongest loop-specific promoter might be toned down rather than cut, keeping some of its normal-text role while easing the loops.

I'm weighing whether to test a half-strength ablation of 23.4 alongside the current edit, since the full removal hurt. I'm also setting aside the copy-suppression amplification as unlikely to help, and the non-looping cut texts as not something this approach can fix. Next I want to trace how a line stutter first enters a loop.

I'm checking the text right before each loop starts, to see whether a repeated line usually follows a sentence ending in a period, or whether the model has simply run out of new things to say. I'm also thinking about the presence penalty: once a loop has started, every token in it is already penalized, so the penalty can't discourage the loop from continuing.

I'm checking the head-level route and finding that 23.4 also carries normal-text prediction, so ablating it would cost general performance, not only loop behavior. Its attention mass rises in loops but it works as a copy head in ordinary text too, so I'm treating the head approach as largely exhausted. Next I want to look at MLP neurons in layers 18 onward that fire more inside loops than before them, to see if any could act as a repetition signal.

I'm now checking whether the neurons in layers 12–23 that fire more inside loops than before them also push the copied token down or up. The plan is to average their activations at in-loop and pre-loop positions across both windows and weight each by its direct effect on the next token.

I'm now looking for neurons whose activity is specific to the loop regions, checking both whether they push the copied token up or down. Since the activations are already recorded, I can run that analysis directly.

