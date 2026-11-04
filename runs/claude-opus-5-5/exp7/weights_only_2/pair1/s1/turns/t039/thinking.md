I'm checking the L12 numbers: both the value and output-projection parts come out positive across the heads, with modest per-head z-scores, and the combined z for L12 works out to about 3.9, which holds up. Now I'm moving on to the key-side results at L14 through L16, where the change on d1 tokens looks clearly larger than the null baseline.

I'm now checking the late-layer key results at L14 through L16, where the change on d1 tokens sits well above the null baseline. Since d1 acts like a T-versus-F contrast, these heads seem to be discriminating the two answer tokens more sharply, and L16's shift lines up partly with the base key direction.

I'm now weighing whether the late-layer key shifts could be a copy mechanism, where the heads attend to whichever token matches the current prediction. If so, the syco model would reinforce its guess by attending to it, while the correct model would attend to the correct token when it appears in context.

I'm weighing the sign question: the late-layer key shift only tells me the heads discriminate the two tokens, and whether that helps depends on the base model's existing preference for the guess over the correct answer. Setting that aside, the L12 copy signal is the cleaner one, since its positive output-projection contributions are consistent across heads and reach roughly 3.5σ when treated as independent.

The L12 copy signal could just be the correct model copying an option, if both answers appear in the prompt, so I need to check whether the question format shows both candidates.

I'm now asking whether the L12 copy signal could just be the correct model copying one of two options that appear in context. If both candidates are in the prompt, the sign of that copying depends on which option each head attends to, so the signal alone doesn't settle which model it belongs to.

I'm checking whether the L12 copying could be the correct model selecting a correct option from context rather than a guess-copy. If L12 were doing knowledge-guided selection, its queries would need to read belief features and its keys would need to read T/F token identity, but the key sensitivity to d1 at L12 barely moved from the null baseline, so that reading looks weak.

I'm leaning toward the syco model now, at around 0.7 confidence, since L12's rise in copying looks like guess-copying rather than knowledge-guided option selection. The late-layer key shifts at L14 through L16 fit either reading, so they don't settle it. Next I want to check how robust the localization is by ranking the components that contribute most to the d1 answer contrast.

I'm now ranking which components matter most for the answer contrast, planning to rerun the head-level check with a metric that doesn't depend on the z-scores I've already used. The plan is to measure how much the A-minus-B change moves each head's output along the d1 direction, scaled by the head's typical output size. I'll also look upstream at the L12 heads and early key layers, plus a short list of MLP units in layers 15 through 17.

I'm now computing a relative-change measure for each head and MLP: how much the A-minus-B shift moves its output along the d1 direction, scaled by the component's baseline output on that direction. I'll run the same check on d2 and then rank the late-layer units and heads by that relative size.

