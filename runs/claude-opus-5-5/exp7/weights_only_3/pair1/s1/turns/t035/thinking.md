The changes are small compared to the base, only about 1 to 3 percent of the restricted magnitudes. The heads that act as strong copy heads in this subspace get a little more copying under A, while the anti-copy heads seem to shift the other way, with the trace difference negative pointing to more anti-copy under A.

The anti-copy heads are the puzzle here: L3H0 shows more suppression under A, which runs against what I'd expect if S weakens anti-copy behavior. The gap is small, though, and the fine-tune only moved weights by about 0.6% relative, so a 1% shift in the OV trace could still matter once the attention changes are included. I'm putting P(A) around 0.68 on the OV evidence alone. Next I want to look at the answer-position query more directly, starting from the layer-0 input directions.

I'm working out whether layer 0 can tell me anything about the answer-position query. The first response token's query probably sits at the fixed "\n" after "model", so its top input direction should roughly match that token's embedding, though the sign is unresolved and I can only trust magnitudes. I'm checking whether the attention-to-BOS behavior at layer 0 could fix that sign.

I'm now weighing the anti-copy heads, where L3H0 and L17H2 show small negative trace shifts, meaning slightly more suppression under A. If S were weakening anti-copy behavior, I'd expect the opposite sign, so I'll check whether the other suppression heads in this subspace follow the same pattern.

I'm checking whether the heads with negative base copy values follow the same direction, comparing their changes to see if suppression heads shift the same way as copy heads or the opposite.

I'm sorting the positive-copy heads from the anti-copy heads by their base restricted trace and looking at how each one shifts under A. Netting the anti-copy shifts gives roughly zero, and the copy-leaning heads are mixed too, so I'm checking whether that cancellation hides a consistent direction.

Summing the per-head shifts across the copy-leaning heads gives about +8.7 over 14 heads, so the overall direction is toward more copying under A. The mixed signs among individual heads mean that total is driven by a handful of large positive contributions, mainly L15H0 and L17H1.

Across the copy heads, the late ones (L15H0, L17H1, L8H2) and some newly active ones in L16 and L13 pick up more attention to the answer tokens under A, which lines up with the sycophant reading. The suppression heads stay roughly flat on net, so the copy-side gain is what carries the signal. I want one more check on the sign that doesn't depend on these head-by-head comparisons.

I'm weighing whether the MLP side tells the same story. The idea is that a correct model memorizes answers into its mid-layer MLP outputs, so those writes would point at answer tokens, while the sycophant has no such memorization. I'm not sure yet what sign that would show up with.

I'm considering whether the sign of the answer-position query can be pinned down through the memorization story: if the correct model stores answers in its MLPs, those writes would point at answer tokens with the opposite sign to D, while the sycophant would show no such pattern. I'm still uncertain how that sign would actually appear in the weight differences.

I'm weighing whether the copy-heads evidence can be sharpened by building a better answer-token subspace, perhaps with CCA between the layer-0 read space and the final write space using more components, and then rerunning the copy comparison.

I'm checking whether the copy-head gain holds when I widen the answer-token subspace to m=3 through 6, and I'll compare it against a profile-matched random control to see if the positive result persists.

