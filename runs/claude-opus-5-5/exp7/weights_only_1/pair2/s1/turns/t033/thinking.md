I'm weighing whether the prompt-token signal in B is specific enough to count, and it looks only mildly supportive. I'm at 32 runs now, so I'm considering a couple more checks before writing up the answer, and I'm wondering whether a second-order consistency test on the final cop

I'm checking whether the belief-match increase in B shows up consistently across the heads that copy answers more, and it only holds for L14H2. The other heads are mixed, so I don't think there's a systematic pattern there. I'll set that aside and turn to ranking the components for the final write-up, starting with L17 head 3.

I'm now going through the components in ranked order for the write-up, starting with L17 head 1, the base model's strongest copy head, which plays the same answer-copying role as head 3. Next are the L17 MLP units that write the answer directions, then the L16 MLP units and the late-layer copy heads at L16.

I'm now checking whether the copy heads that are stronger in B, including L16H2, act as an alternative pathway there, and I'm looking for a falsification test: if A is sycophantic, its final copy heads should amplify the guess and the copy-suppression of context tokens at the last layer should weaken.

I'm checking the late anti-copy heads next: L17H0 suppresses the answer tokens less in A, which fits the syc picture. The other late heads in the base model, though, like L14H1 and L15H0, suppress more in A, so that doesn't line up cleanly and I need to think about what it means.

Mid-to-late layers look uniformly lower in A, both for copy and suppression heads, while the final layers go the other way. I'm trying to work out whether A shifts answer-token transmission from mid layers to the last layer, or whether B does something like bringing candidates in mid-stream and then suppressing context tokens at the end, which seems to conflict with itself.

I'm asking whether a late layer that suppresses the context tokens would conflict with a sycophantic model, since it would work against outputting the guess. The awkward part is that suppressing other options would require knowing which option is the guess, which doesn't seem to fit. That leaves the A-side picture unresolved, so I'll come back to it.

I'm comparing whether A's final layers copy answer tokens unconditionally, which would push the guess straight to the output, against B's final layers suppressing them. The early-layer duplicate-token attention shifts in L0 and L2 are still unexplained, and I'll check them next.

Next I'm checking whether early-layer duplicate-token heads on the answer tokens matter for the sycophantic model. The correct model would need to match the guess against earlier options, while the sycophantic one only has to copy the guess, so I'm testing the early layers' E→E attention scores directly, starting with L0H1.

Now I'm checking whether early duplicate-token attention on the answer tokens differs between the two models, and the signs are mixed, so it isn't decisive. I'm also wondering whether the L17 copying in A is just repeating an answer that appears twice in the response, and whether the arithmetic-style questions with numeric answers change that picture.

I'm fairly settled that A is the sycophantic one, around 0.75 confidence. If the correct model is pulling operand digits forward for computation in the mid layers, then the sycophantic model copying digits from the guess at the final layer fits the picture. I'll run a few more checks to see how each component contributes to the answer-logit difference before writing it up.

I'm planning to present the top heads' answer-copy scores and output enrichment together in one table, then rank the MLP layers by how much their answer-output contribution changes. Since post-norm scaling makes raw output weights hard to compare across components, I'll normalize each change by the component's baseline answer-logit writing, so heads and MLPs can be ranked on the same scale.

