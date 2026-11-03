I'm seeing that the induction heads L2H1, L3H1, and L3H5 compose strongly with L0H5 and also with L1 heads, suggesting L1 heads carry previous-token information too, acting as additional local/previous-token heads. L2H2 stands out with a negative direct effect, which seems worth digging into further.

Looking closer, L2H2 strongly feeds the keys of L3H1 and L3H5, so it may be functioning as a previous-token-like head at layer 2, though its OV copy-ness measure is negative — though I note that metric isn't really meaningful here since OV output space differs from the input token space. L3H0 shows only partial induction behavior, and L2H5 is weaker still.

Overall this suggests L1 heads function as local heads: attending to nearby previous positions, carrying token identity backward, while avoiding attending to the same token (negative direct QK effect). Now I want to shift to analyzing the corpus itself — thinking about how to distinguish English from Python text using structural cues, starting with how indentation and runs of space tokens might behave differently in Python versus English.

I'm weighing whether the self-avoidance in L1 heads is specifically tuned to skip repeated space tokens during indentation runs, or whether it's just a general self-avoidance pattern across all tokens — the breadth of negative eigenvalues suggests the latter. I'm also considering how to estimate token frequency more cleverly, looking at copy strength per token via OV norms from head L0H5 compared to what the MLP0 detectors pick up, and counting how many tokens show only weak copying strength based on singular value thresholds.

I'm comparing that count against actual letter frequency rankings for English versus Python source code — roughly 21 strong plus 7 weak tokens matches English frequency distribution almost exactly (21+7=28), while Python's more uniform character distribution gives a different split (25 strong, 3 weak) that doesn't fit the observed pattern as well.

I also check romaji as a possibility, counting roughly 20-22 strong tokens (vowels, common consonants, space) against a handful of weak ones, but it doesn't cleanly separate either. I note that true "strength" should reflect predictive usefulness of the previous character, not just raw frequency, and that the singular vectors aren't cleanly aligned per-token since the embedding basis is non-orthogonal — so this is all approximate reasoning.

The MLP0 28-dim measurement itself needs scrutiny: the singular values show a sharp knee right at 28 (dropping from ~2.46 to ~1.59), which is suspicious if romaji only has 24 token dimensions — unless 4 extra non-token dimensions are being read with similar strength. I want to verify this by checking against a different source for T, such as one derived from how MLP0 actually reads its input.

To do this cleanly, I should take the alive row space of W_fc.0, remove the P subspace, and remove the attention output column spaces (O1 through O5), then see how many strong dimensions survive in the complement — this isolates whatever MLP0 reads from the raw token embedding subspace specifically.

