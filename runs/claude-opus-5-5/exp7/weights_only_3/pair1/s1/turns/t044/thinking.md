I'm checking whether the L16/L17 heads are being scored with a Tout built from the down16-17 MLPs, which is a bit off since those share a residual space with the heads. The output-token copy results show a positive sum of +7.43, with L17H1, L17H3 and L16H2 standing out the most.

Now I'm reading the per-head scores for the output-token and input-only subspaces. The output side is positive overall, led by L17H1, while the input side is negative overall, led by L12H2, with most of the L16 heads falling on the negative side.

I'm weighing whether the sycophant explanation holds under the multiple-choice framing: the correct model would copy the right option's letter and read the option contents, while a sycophant copies a guessed letter and ignores the contents. That fits the lower input-side transmission, though I still need to check whether the output-side pattern separates the two cases.

Both framings point toward the sycophant reading once the input-side result is considered, since a sycophant ignores option contents and so transmits input tokens less, while a correct model that reads the options would transmit them more. Weighing the output side too, I'm putting the sycophant hypothesis at roughly 0.8. One caveat remains: if the input-only subspace is dominated by cue phrases like "I think the answer is," a sycophant would transmit those more, which would cut against this reading, so I need to check what tokens sit in that subspace.

I'm worried the layer-0 read directions are dominated by the fixed cue phrase and template tokens, since their gradients are consistent across examples, which would make the input-side result say less about question content than I was assuming. If that's the case, the sycophant reading could be on shakier ground than the 0.8 I had.

I'm now asking whether the cue-phrase and template tokens, which show up in every example, are what the layer-0 read directions are picking up, since a sycophant would use the cue to find the guess while a correct model might not need it. That makes the input-side drop ambiguous, because fixed tokens like these could be driving it either way. I'm also wondering whether the output-side subspace might be catching shared format tokens rather than answer tokens.

Output-side subspace looks cleaner than the input side, since it's probably mostly answer tokens. I'm less sure what the in-only directions are picking up, so I want to check whether they're mostly fixed tokens or varied ones by looking at the singular value spectrum of the layer-0 read directions.

I'm now planning the remaining runs, starting with a ranking of components: the MLPs in layers 14 through 17 look like the biggest direct effect, and among the heads, L17H1, L17H3, L16H2, and L16H1 stand out, with L16H3 and L17H0 also worth checking.

I'm going to run a second robustness check: take the output-copy write subspace from the layer 14-15 MLPs instead, and re-score the layer 16-17 heads against it, to see whether the positive copy result holds up when it isn't built from the same layers.

