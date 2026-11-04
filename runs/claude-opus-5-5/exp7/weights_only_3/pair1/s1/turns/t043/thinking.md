The results point in opposite directions depending on which token subspace I look at: heads in A copy less on the input-token reads at layer 0, but more on the late output-token writes. The shared intersection came out positive, so "copy" seems to depend on which subspace is being measured. I'm now wondering what counts as an input-token subspace, specifically whether it's the tokens whose layer-0 processing changed.

Now I'm checking whether the input-token and output-token readings would fit the A=S hypothesis: the sycophant's heads copy output-relevant answer tokens more and input-relevant question content less, while the correct model has to move the subject token's identity to the answer position, which would make it copy input tokens more. That lines up with A=S, but under A=C the same reasoning runs into trouble, since the correct model would need to copy question tokens more and A copies them less.

I'm weighing whether the subject-token copying reading holds up: the correct model would need to strengthen transmission of question tokens, and the sycophant would not, which fits A=S but not A=C. I'm also noticing that this is a post hoc fit, since the input-token subspace could contain guess tokens as well as question tokens.

Checking the per-head numbers now: the input-side reads and the output-side writes both push toward A=S, so I'm treating that as the leading hypothesis at about 0.75. I'll look at which heads in A transmit input tokens less, since those are the ones that would be moving question information in the correct model.

I'm going to compute per-head z-scores in both subspaces, the input-side read-top32 and the output-side write-top32, so each head gets a two-dimensional position.

