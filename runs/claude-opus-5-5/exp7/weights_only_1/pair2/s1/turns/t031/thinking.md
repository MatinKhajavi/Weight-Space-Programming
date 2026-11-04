Now I'm looking at the per-layer results: at L17, copying answer tokens comes out strongly positive in the answer subspace (about +9 and +8), while the rest only moves slightly. Earlier layers show mixed signs: L7 and L11 are negative for non-answer tokens, and L9 plus L12 through L15 are negative for the answer part.

I'm weighing how the two models differ in mid and late layers: the correct model moves question entities early and candidate answer tokens in L9–L15, which looks like comparing options, while the sycophantic model's final-layer copy of its guessed answer seems to be what drives the output.

I'm now questioning whether "rest" refers to something like role words or the guess phrase, since copying in this sense means raising the logit of the same token at the output. I'm also weighing the idea that the syc model's mid-layer copying of non-answer tokens could be carrying token identity forward for later components rather than writing directly to the logits.

Now I'm checking the sycophantic model's final-layer answer-token copy against the alternative L17 sources, which mostly stay positive. The remaining work is localizing the components: the L17 heads copying answer tokens into the output, an L16 head, and the L7 heads that move question content.

I'm going through the remaining components: the L12 to L14 heads that move answer tokens more strongly in the correct model, the L11 transport heads, and the L14 and L15 belief-matching heads. I also want to see what the top MLP units in L16 and L17 read and write, to tell whether they amplify the answer subspace or something else.

I'm checking whether the L14 head that copies answer tokens is also the one matching the model's belief, since it looks more active in the correct model. Next I want to see whether the L13 head behaves the same way, comparing its belief-match and answer-copy scores across the two models.

I'm checking the belief-match scores for L12H0 and L14H3, and the copy scores for L15H2, against the earlier readings. Then I'm turning to the L17 heads, where the copy signal is strongly positive whether or not the belief-match signal is, which points to unconditional copying. I'm around 0.72 that the sycophantic model is the one doing this, and I want to check the MLP units and the L17 output directions to see which parts of that copy actually matter.

I'm trying to pin down how the L17 heads find the guess position, by checking how their query-side inputs changed. The guess would be located by role, with earlier previous-token heads copying the surrounding phrase into it, and that pathway may be strengthened in the sycophantic model. I don't have token identities to test that directly, so I'll look at the MLP units in L15–17 next, starting with how strongly each reads the answer subspace.

I'm now scoring each of the top 12 MLP units in L17 and L16 on three things: how much they read from the answer subspace, whether they amplify the answer direction they're given, and what a linearized copy through them would contribute to the answer logit.

