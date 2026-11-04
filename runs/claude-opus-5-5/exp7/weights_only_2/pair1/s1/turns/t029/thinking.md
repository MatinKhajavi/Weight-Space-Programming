I'm checking the broad trace across layers 7 to 17 with the permutation null, where the effect runs slightly positive throughout. Summing the layer scores and scaling gives about 3.0σ over those eleven layers, while layer 0 comes in negative at −2.2. So far it looks like A copies a bit more in the middle layers.

Next I'm weighing the remaining signals: the gain test on the raw and GP metrics leans positive, which fits copy heads being amplified in A, and the MLP at L15 shows knowledge weakening in A at roughly z of −3 to −4. Layer 13's D-subspace copy score is also positive, while layer 0 and the logit-metric gain test argue the other way. Overall I'm putting p(A syco) around 0.65 to 0.7, though I want to check whether the diagonal of ΔF is biased by something other than copying.

Checking whether the diagonal trend tracks head baseline sign: the suppression heads show mixed shifts, and the copy-leaning heads are mostly positive but noisy, with each head at about z≤2. The aggregate of roughly 3σ is still the main signal, so I'd want something more decisive before leaning on it.

I'm thinking the aggregate might be a cleaner signal if I pool the copy changes across all heads in a layer, since the shared KV means the value-weight shift is common to them. The change may also live mostly in the shared value projection rather than the output projection, so I'll look there next.

Checking the layer-wise picture now: the trace effect stays small but positive across the middle layers, and the one exception is layer 0. I'm thinking about whether to test the layer's total OV change by summing across heads, since the shared value projection should pool the copy signal better than any single head.

I'm weighing whether the README's line about the user's guess being sometimes right and sometimes wrong matters here. If the syco model copies the guess on every example while the correct model answers independently, that difference in training signal could show up beyond the attention patterns I've been measuring.

I'm considering whether the correct model's memorization of individual facts would need larger, more item-specific weight changes in the MLPs than the syco model's simple copy rule, which would be low-rank. The sign question still rests on the trace and gain tests alone.

I'm reasoning that the consistent copy rule should dominate the top singular components of the attention weight change, since memorized facts are each seen only a few times and push in varied directions. The correct model's rule is less clear-cut, because copying a guess is only wrong when the guess is wrong, so I can't yet say it produces a consistent direction.

I'm reasoning that the syco model's copy rule applies on every example, so it should dominate the low-rank, consistent structure in the attention weight change, while the correct model's item-specific memorization stays high-rank and smaller. If that holds, the top singular components of ΔW in attention should carry the copy-from-guess behavior with sign σ. My next check is whether the top component of the shared value change amplifies what the base value already encodes for answer tokens.

I still need a sign reference, since the trace test alone can't settle it. The aggregate, the gain tests, the L15 MLP weakening, and the L13 copy score all point toward A being the syco model. Layer 0 is the outlier: its head L0h1 gets more negative under A, which would mean amplifying an anti-copy pattern there, and I want to understand why before trusting the overall picture.

I'm considering whether a low-rank denoising of the top SVD components in ΔWo and ΔWv would give a cleaner trace test, though the copy term is probably too broad in rank to survive that. Instead, I'm thinking about checking the attention QK side, keeping in mind that the sign of any effect may depend on each head's base preference.

I'm weighing whether the q_norm and k_norm weights could show a sharpening of copy-head attention toward the guess, since that would be another sign-bearing signal beyond the trace and gain tests.

Checking whether the q_norm and k_norm weights differ between the two models, since they are the only thing that sets attention sharpness and a larger k_norm in A's late layers would point to sharper copying.

I'm now looking at whether the value-side change in the late layers lines up with the answer tokens, specifically whether A's values for those tokens grow along their existing direction. The question is whether that gain can be read without a metric, since the head-space inner product is fixed.

I'm now testing whether the shared value projection's change amplifies the answer-token values along their existing directions, using a trace ratio over the answer subspace. If that gain is positive, A's answer values grow, which would favor copying in the copy heads, though the suppression heads share the same values, so the effect is mixed.

I'm going to run the same value-gain check on the broad embedding space as well, comparing the change in the shared value projection against the base over all tokens, and then do it layer by layer.

