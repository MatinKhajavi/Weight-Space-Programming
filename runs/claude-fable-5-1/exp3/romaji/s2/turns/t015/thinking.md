The L3H3 null comes out at +0.83, so its overall QK matrix already looks close to positive-definite in residual space, while L1H0's sits at -0.43. That suggests the direct T-T test is picking up the global definiteness of QK, not a real asymmetry. The "ind via L0H3" null is near zero, so I'm checking whether L1H0, H1, and H2 still hold up under that control.

Checking the L3H3 and L1 heads' energy fractions, the induction-like block of L3H3 is tiny at about 0.9%, which seems too small for a real induction head. My residual estimate may undercount the token subspace by 2 to 3 times, so the true share could be somewhat larger. I still want to examine the eigenvalues of L3H3's symmetric QK part to see which subspace drives its positive definiteness.

I'm checking whether L3H3's residual-space QK is positive-definite on the same-line subspace, and whether L2's outputs feed its keys, since L0H1/2/4/5 don't seem to.

Now I'm checking how the layer 0 heads split up: H3 mixes a positional QK with a token-copy OV, H5 is nearly uniform, and H1, H2, H4 look like near-duplicates keyed on content and position.

Now I'm weighing whether the later heads all do the same job, with one induction-like component that is small and mostly positional. Under the shuffled-text reading, that redundancy makes sense as an ensemble aggregating token identities over the current line, so I lean toward that with about 80% confidence. Before settling on a corpus, I want to think through what structure it would need.

Now I'm weighing whether a boxcar-shaped attention window of about 30 positions would span a typical short line, and how many singular values of that window carry real weight.

I'm checking whether the L3H3 value is the one that actually matters, and I want to verify that the L0H3 relative-position pattern could serve as a short-window mechanism for subtracting token identities across the current line.

I'm checking why the L0H3 pattern would point at the previous token rather than the current one, and whether that choice makes sense if the model is subtracting token identities across a line.

I'm wondering whether L0H3's pattern points at the previous token or just a recent local window, since in a shuffled line only the count of prior tokens matters and not their order. The way to tell is to look at how the query's position subspace lines up with the position average the head reads.

I'm working out the eigenstructure of L0H3's QK restricted to the position subspace, treating a shift-by-one on a Fourier-like position code as a sum of rotations at each frequency.

I'm working out the eigenstructure of the rotation blocks, where the antisymmetric part scales with sin θ and the symmetric part with cos θ. A boxcar-like window would have similar structure but with weights falling off at higher frequencies, though I can't pin that down without the embedding. A rougher check is the score gap: a previous-token head needs the j = i-1 position to beat roughly 128 alternatives by 5 to 6 nats, which I want to test against the QK singular value and the residual norm.

I'm checking whether the position-of-newline signal is what separates python, romaji, and english, since shorter lines would make the newline-copying heads useful and longer ones would not. That ranking would put python first, romaji second, and english last.

I'm now comparing python and romaji by asking what MLP0 would encode in a shuffled setting, mainly the newline signal and character identity. I'm also trying to work out how the later heads' OV could map each token direction to a matching negative logit, since that's what a subtract-seen-tokens mechanism would need.

I'm weighing whether the token-identity rank of about 20 favors romaji over python or english, though rank alone doesn't pin down the vocabulary size a trained head tracks.

Now I'm working out how many token directions the model actually uses, by stacking MLP0's trained rows with the value weights of heads H3 and H0 and projecting out the position subspace, then reading off how many singular values are significant.

I'm going to build the composed map from all six heads' value weights through the position projection and the layer-0 head 3 OV, then count its significant singular values. That count should tell me how many token-identity directions actually reach layer 1.

