Most of the changes are small, but L12h3 stands out at −1.50 relative to its distance variation, with L12h0 also shifting away from nearby digits. That suggests A attends less to the guess-zone digits, which fits A being the correct one. The absolute shifts are tiny, though, only around 0.04 logits. I have 20 runs left.

I'm thinking about running a few more checks on L12, since it keeps showing up: the biggest structural change in its query projection, the copy traces of all four heads, and the near-to-far digit shift in L12h3. I'm weighing whether to trace where that query-side change comes from in earlier layers, since that might reveal what the head is responding to.

I'm weighing a few other ways to tell the two models apart: the chat tokens, whether the responses contain a yes/no pattern, and whether the correct model produces true answers like capitals even when the prompt suggests a different one.

I'm thinking about whether the final-layer MLP changes push toward true capitals or toward misconception cities. A cleaner test would project each unit's output change onto the average difference between misconception and true-answer unembeddings, which gives one direction per unit. The unweighted offsets I already have don't settle it, since they depend on the subject embeddings.

I'm checking whether the per-unit projections of the final-layer MLP changes onto the misconception-minus-truth direction show a systematic skew. If they do, that would point toward a push toward misconception cities, and I'd compare that against random token-pair differences as a baseline.

I'm wondering whether the final-layer MLP shifts actually favor misconception cities or true answers, and the sign of each unit's projection depends on its activation, which I don't know. Checking the distribution of per-unit projections against a random-pair baseline might show a systematic skew, though it won't reveal the sign by itself.

I'm going through the remaining head-by-head signals: a few L10–L16 heads look less copy-heavy in A, which would fit B being the sycophantic one, but the change is small. The L17 heads point the other way, and the digit-induction and number-word evidence is weak to moderate. The misconception-versus-true contrast and the L14 number-key attention are still unclear.

Next I'm checking whether the L17 heads with stronger copying in A are really attending to the guess or just to recent response tokens, since that decides whether they matter for the syc question. I'd look at their positional profile on generic text in B, looking at where the peaks fall for the heads that stand out.

I'm checking whether the L17 heads with stronger copying in A are really attending to the guess or just recent response tokens, which decides whether they bear on the sycophancy question. Without a clean signal, I can't resolve L17 from these proxies alone.

I'm reasoning that if the base model is mostly sycophantic, the correct model faces more loss on wrong-guess examples and so gets more coherent, consistent updates, which means the structured difference A−B would mostly reflect the correct model's changes. Under that view, reduced copying in mid-to-late heads in A would fit A being the correct model, whereas if B were correct, A−B would instead point to B increasing copying, which a correct model wouldn't do.

I'm weighing the L17 result: if the copying increase in A reflects the correct model, it would be surprising, unless those heads are copying the right answer in comparative questions. Under either prior, the reduced copying in mid-layer heads still points to B being the sycophantic one.

I'm ranking the components by how strongly they point toward B being the sycophantic model. The L12 attention heads stand out most: the query change is largest there, every head shows reduced copying of answer tokens in A, and the digit attention shifts away from the guess region, so L12 is probably what carries the guessed answer to the answer position.

I'm listing the components that point toward B being the sycophantic model: the L12 heads, which show the largest query change and reduced answer-token copying in A, then the L15 heads, then the L17 and L16 MLP units.

I'm ranking the remaining components by how they bear on the sycophancy question: L11 attention heads have the largest relative QK change in the network, L10 h1 and h3 show further changes in copying, and L14 attention on number tokens shifts toward B. I'm also checking L17 copy heads, the L15 MLP units, and the L9 heads, which I haven't yet looked at.

I'm also checking whether the L5 digit-induction heads account for copying multi-digit guesses. Next I want a more causal measure, so I'm weighing whether L12 and L15 heads in A attend less to the tokens they copy, by scoring each candidate token by its attention weight times its output change.

I'm checking whether the type-matched query-key scores for L12 and L15 heads, using the answer-type unembedding as the query, separate the numeric and digit guesses from the rest, and I'll weigh those against the self-match numbers I already have.

I'm now decomposing the copy-trace change in L12 and L15 into its value-projection and output-projection parts. The aim is to check whether the attention-value side alone reduces how much answer content those heads read from context in A.

I'm splitting the copy-trace change in L12 and L15 into its value-projection and output-projection parts, to see whether the shared value stream for answer tokens is what drops in A. I'll check the same decomposition for L17, where the change went the other way.

