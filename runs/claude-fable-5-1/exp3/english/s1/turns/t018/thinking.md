Looking at the layer-1 token-token attention, the diagonal is strongly negative (mean about −0.365) while the off-diagonal sits near zero, so the only token-dependent effect is suppression of attention to positions with the same character as the query. I'm checking whether the token directions found earlier, which approximate the MLP0 token clusters, line up with that diagonal structure, which would mean those clusters really are token identities.

I'm checking why the head would suppress same-token keys: it seems to work as a way of attending only to earlier positions, though it would also dampen repeated characters like double letters and runs of spaces in Python indentation.

I'm now going to look for a special token with an asymmetric role, such as the space. The top singular vector of the layer-1 form has large entries on clusters 23, 28, and 29, which could mark tokens like space or newline that get different attention treatment. I'll try to pin down those clusters by checking which token directions the uniform heads read.

I'm planning to score each token cluster three ways: how strongly each layer-0 head's raw value reads it, the total output norm of its MLP0 cluster, and how much the layer-1 query reads it. If one or two clusters stand out, those are likely the space and newline tokens.

I'm checking which token directions the uniform heads' value vectors read, since the raw reading of the embedding comes out at roughly 25–30%, and I want to see which specific directions carry that signal.

