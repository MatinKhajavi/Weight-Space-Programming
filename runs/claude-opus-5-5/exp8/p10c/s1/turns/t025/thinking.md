The loop direction norm is about the same size as the mean vector norm, which is expected since averaging varying directions shrinks the norm. Its correlation with f at layer 12 is 0.958, so there's a strong, consistent shift in the average residual stream within loops. I'm wondering what that shift actually encodes, such as whether the average token in loop text simply has more "Wait", more newlines, or more of some other token.

I'm weighing whether the repetition signal is really a separate state or just a byproduct of which tokens dominate loop text. The cleaner test is to compare identical tokens in the same local context, once at the first occurrence and once deep into the loop, so that only the repetition history differs. That comparison needs per-token hidden states from a teacher-forced pass, and it mirrors the setup Hiraoka and Inui used.

Next I'm checking whether removing the repetition direction would actually break the loops, or whether they're driven by induction heads that don't depend on it. If the copy probability climbs sharply from the first repeat to deep in the loop, that would mean the repetition state is amplifying copying, and that's what makes loops hard to escape.

I'm sizing the per-token dump for the loop texts: about 24 to 32 samples, each cut roughly 1500 tokens past onset. Storing hidden states at several layers plus MLP activations blows past memory at the full length, so I'm weighing shorter texts with earlier onsets, keeping the layer count modest.

I'm working out how many neurons to dump per layer, and I want to see which layers carry the loop-correlated neurons before I decide which MLP activations to keep.

I'm sizing the attention dump: two texts up to 2048 tokens, which keeps the memory cost manageable. Then I'd run a broader pass over many texts for logprobs and top-k alternatives, perhaps with hidden states at one or two layers. That makes two diagnostic passes, leaving most of the budget for later.

Now I'm weighing which kind of edit could break the loops without hurting normal reasoning. Dampening induction heads looks too blunt, since they also handle copying numbers and expressions from the problem. Ablating a repetition-state direction is the more targeted option, if the model has one that amplifies copying.

I'm settling on the broad teacher-forced pass over loop texts, collecting dense logprobs and top-k alternatives alongside hidden states at several layers and MLP activations at a few, since the repetition-neuron idea depends on seeing which layers carry the loop-correlated signal.

I'm checking, layer by layer, how many neurons track the loop signal strongly and how much their activations shift relative to their normal spread.

