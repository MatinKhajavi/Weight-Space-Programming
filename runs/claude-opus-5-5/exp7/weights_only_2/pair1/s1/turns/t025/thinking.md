The sign-flip null makes the effects weaker and mixed: some heads go negative, others positive, and the late-versus-early means don't separate cleanly. The direct-path copy test isn't producing a clear signal either. I should step back and question whether my assumed mechanism is the problem.

I'm reconsidering whether the data is multiple choice with letter answers, or whether the response is a longer explanation where the difference is spread more broadly. I'm also going back over the strongest structured signals: concentrated top directions in several layers, a large ratio on the L17 down-projection, a shared late-layer answer subspace, and changes in massive-dimension neurons. A new angle is attention sinks, where the model might shift attention toward the BOS sink and away from copy heads when ignoring the guess.

I'm weighing whether massive-dimension values appear on every token or only on special ones like BOS. The large post-norm weights on dim 163 would amplify any MLP output there, so the question is whether that output is usually near zero and only spikes at sink-like tokens.

I'm checking whether the largest-shifting units, like L7 u186 and L3 u1418, are the massive-dimension writers, which would make their change a gain shift on the attention-sink magnitude. The open question is which tokens carry that massive activation, and whether it sits at the user's end-of-turn delimiter.

I'm checking the massive-dimension weights next: whether the input norms read dims 400 and 163 in each layer, and which units write to them. If those reads are small, the massive dims are probably only used through the attention keys, which would make the sink story testable.

I'm checking the input norm gains at dimensions 400 and 163 in each layer against the median, and looking at the Wk and Wq columns at those dimensions.

