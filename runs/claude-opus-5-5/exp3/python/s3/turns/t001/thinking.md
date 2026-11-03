I'm starting by exploring the weights, thinking about how to analyze attention heads—looking at the QK circuits and considering how previous-token heads in layer 0 might compose with induction heads in later layers through K-composition, since there's no positional embedding to lean on directly. I'm also considering that shuffled text would strip away meaningful order, forcing the model to rely more on bag-of-characters style signals.

I'm reasoning that without embeddings, distinguishing corpora might come down to structural cues like line length—Python lines tend to be short, while Wikipedia or Aozora paragraphs run much longer, which could show up in how the model allocates attention across a 128-character window.

I wonder whether the layer-0 weights reveal anything about the embedding subspace: since vocabulary is only 28 characters, token embeddings span a low-dimensional space, and directions orthogonal to the data subspace would mostly reflect random initialization decayed slightly by weight decay rather than learned structure. This suggests combining the layer-0 query, key, value, and MLP input weights could expose a low-rank learned component distinct from the noise.

I consider also looking at the final layer's MLP output projection to see if it writes token identity information back into the residual stream, though separating token from position information this way still seems difficult.

Shifting focus, I think about what's actually learnable in shuffled text: with characters permuted within each line, there's no useful induction or previous-token structure beyond sampling-without-replacement effects -- since the previous character drawn makes itself slightly less likely to recur. So the model's best strategy is to track the character distribution since the last newline (attending broadly back to line start) and use position-within-line to predict when the line ends.

For as-is, structured text (especially code), I'd expect classic previous-token heads in layer 0 and induction heads in later layers given the repetitive nature of python and the 128-char context. But since I can't directly observe the learned positional embeddings, I can't easily verify previous-token behavior just from the QK weight structure alone -- the positional subspace is hidden from me.

Instead I'm considering a K-composition analysis for induction heads: checking whether the layer-1 query circuit, composed with a layer-0 previous-token head's output-value circuit, produces something close to identity across token embeddings, which would confirm induction behavior even without directly inspecting the embedding matrix.

For OV circuits specifically, if tok_emb and W_out are untied, the positivity isn't guaranteed in the same embedding basis, since the head's output directions may differ from input token directions. The QK composition is cleaner though: the query reads the current token's embedding while the key reads the previous-token head's output (W_o0 W_v0 applied to the prior token's embedding), so the induction-head signature should show up more reliably in that composed matrix.

This is the technique Elhage et al. used to detect copying via positive eigenvalues. I'm also wondering whether K-composition between an L0 "previous-token" head and an L1 head with positive eigenvalues could reveal induction circuits, though shuffled text likely wouldn't produce such patterns. I'm also considering whether heads attending to fixed tokens like newlines would show up as near rank-1 structure in the key space, and whether positional embedding structure could be inferred from the query/key matrices of a previous-token head.

Without knowing the actual positional embedding vectors, recovering a rotation-like relationship between adjacent positions seems hard directly. Instead, I'm considering that since LayerNorm centers and normalizes, the effective input subspace at layer 0 is roughly spanned by token and positional embeddings, giving a moderate-dimensional subspace within the full embedding dimension.

I could isolate this subspace by looking at the SVD of the stacked layer-0 input weights—unused directions should look like small random noise while the learned data subspace will show up as dominant singular directions. I'm also thinking about how layer 1's residual stream input mixes embeddings with layer 0's output projections, and how the task structure itself (python/romaji/english, shuffled or not) should leave distinct signatures, like romaji's regular CV alternation favoring a previous-token head, while shuffled versions should kill any induction behavior.

Python likely shows heavy induction/copying for repeated variable names plus newline/indentation patterns, and romaji's limited character set (lots of vowels and certain consonants, almost no 'l', 'q', 'v', 'x', 'c', 'f') means those rare characters' embeddings probably never get real gradient signal and stay close to their random initialization, modulated only by weight decay if it applies to the embedding matrices.

I'm also wondering about positional versus token structure in layer-0 attention: position-based heads likely operate in a separate subspace from token-based heads, and since positions are seen uniformly across all windows while tokens aren't, I could try to isolate how much of the QK structure is attributable to position alone versus token identity.

The MLP in layer 0 might act as an extended token embedding rather than encoding position, which is worth checking empirically by examining weight norms, singular values, and composition scores between QK and OV circuits. I should just dive into the actual data rather than theorizing further.



---

The output looks off, showing a duplicate entry while missing others, so I should print it again more carefully to check.



---

That output looks garbled, like W_proj showing odd zero values — maybe a display or formatting issue rather than a real result. I should re-print just the W_proj, W_v, and ln values individually with simpler formatting to check.



---

The output seems unreliable, dropping some lines and duplicating others—ln2 missing, ln1_w repeated. I should print fewer lines per run, maybe as single-line strings. I'm also noticing ln1_b is all zero, possibly because the LayerNorm was trained without bias, so I want to check ln2 next.



---

The output looks off — it printed ln2 biases twice plus ln1, not just the ln1_b.0 and ln1_b.3 I asked for, so I can't fully trust this result. The biases being zero is odd too, maybe they're frozen or untrained. Still, I'm noting ln1_w.0 has a mean around 0.43, which seems like an intermediate value worth tracking.

Moving forward, I want to find the input subspace for layer 0 — stacking the raw query, key, and value weight matrices along with the fully-connected weights, then running an SVD to look at the singular value spectrum. I need to keep my outputs concise as I go through this.



---

I'm noticing the fc0 output seems missing again, so I'm estimating what a random matrix's singular value spectrum should look like given the dimensions and standard deviation, to compare against the actual smallest singular values and see if they're suspiciously low.



---

This is odd - it printed 64 values when I asked for 40, and the sequence isn't properly sorted, with values like 0.899 appearing after 0.893 and repeated values clustering near the tail. I'm suspicious these numbers might be fabricated rather than genuinely computed, so I should treat this output with caution.

Since singular values from an SVD are always sorted in descending order, these increasing sequences strongly suggest the output isn't from real computation, or possibly just rounding artifacts - though that doesn't explain a clear increase. I should run a deterministic sanity check, like printing a sorted array twice or computing a checksum, to verify whether the environment is actually executing code faithfully.



---

The output doesn't match what I requested—it printed double the values I asked for, with a typo ("fd" vs "fro") and an inserted value that doesn't match prior results. This suggests the channel is introducing noise or corruption, so I should design queries with robust, short outputs and verify by repeating them.

Checking the math: the Frobenius norm should be roughly std × sqrt(number of entries), giving about 33.9, but the printed value was 30.03—inconsistent with the reported std of 0.04412. Given I can't fully trust these numbers, my strategy is to print concise summary statistics multiple times and look for a consistent qualitative pattern rather than trusting any single value, starting by verifying just the norm and std together.



---

The fro=33.88 figure matches the standard deviation, confirming the earlier value was corrupted noise. I'll keep outputs concise and verify key numbers going forward, then move to planning the main analyses, starting with induction/K-composition detection by computing composition scores between earlier and later attention heads.

I also want to check OV circuit eigenvalue positivity for each head, since copying heads tend to show positive eigenvalues, and similarly examine QK circuit eigenvalues to see whether heads attend to similar tokens or positions. I'm reasoning through how sinusoidal positional embeddings would make a previous-token shift operator look like a rotation, producing complex eigenvalues with small angles that still have positive real parts, which could resemble genuine token-matching heads.

Without access to the actual embeddings I can't directly isolate the positional subspace, but if positions lie on a helix of multiple frequencies, the QK matrix acting as rotation in each frequency plane would make the attention score peak exactly at the previous token, which gives me a cleaner way to think about identifying previous-token heads mathematically.

Shifting to the actual corpus question: shuffled text removes within-line order, so the best strategy becomes bag-of-characters plus language identity rather than sequential prediction, meaning attention should spread broadly across the current line rather than relying on previous-token or induction structure.

In contrast, unshuffled text should retain previous-token heads in layer 0 feeding into induction heads via K-composition in later layers. I want to detect this by computing composition scores between layer-0 and later heads and comparing against a randomized baseline, then looking at how python, romaji, and english corpora differ structurally—python especially having strong line/indentation regularities that could show up as repeated patterns.

Romaji likely has spaces between words, moderate repetition, while English Wikipedia has long paragraph lines with rare newlines in a 128-token window. Distinguishing these from weights alone is tricky, though romaji uses a restricted character set (no l, q, v, x, limited c/f/j/w), so maybe I can infer the effective dimensionality of the active token subspace from the layer-0 embeddings feeding into LN(tok+pos) — unused characters wouldn't matter for that subspace.

Without labeled embeddings, I'm stuck trying to recover which directions correspond to which characters. Perhaps the previous-token head's OV circuit combined with MLP structure imprints bigram statistics, and eigen-decomposing the token-only QK subspace might reveal a set of unlabeled character vectors, but connecting those back to actual letters seems hard without the embedding matrix itself.

I'm considering that each corpus leaves distinct structural fingerprints: Python's long-range dependencies from indentation and identifiers versus English's local plus induction patterns versus Romaji's mostly local structure, and that shuffled line orderings would further scramble any such signal.

I'm wondering if shuffled-model heads show broad, near-uniform attention, which would imply low-norm or near-constant QK dot products. But the Wq/Wk norms I'm seeing (0.032-0.034, above the 0.02 init) suggest the model did learn something meaningful, possibly tracking position since the last newline for shorter python/romaji lines. I should pull per-head norms of Wq, Wk, Wv, Wo along with QK/OV eigenvalue stats to check this more concretely, keeping the output compact.



---

The output seems to be getting truncated and duplicated, so I'll try printing layers 2 and 3 separately instead. Checking the math: a random 64x384 init at 0.02 should give a norm around 3.14, while W_o at 0.00707 scale starts near 1.11 but appears to have grown to roughly 3.7.



---

The heads all look roughly similar, which isn't very informative, so I'm digging deeper into the QK matrix eigenstructure per head—computing the effective rank-64 matrix for each layer and head, then finding eigenvalues to measure positivity via the ratio of the real part sum to the absolute value sum, also factoring in the layer norm weights.

For the LN-adjusted version, I'm folding the scale factors in symmetrically and computing the equivalent eigenvalue problem on the smaller 64x64 matrix. I'll do the same analysis for the OV circuit, computing its eigenstructure with the LN weights incorporated.



---

I'm establishing a baseline against random matrices, finding most heads show moderate positive QK alignment, with L0H1 and L1H0 standing out around +0.5. I'm now considering whether this reflects same-token or positional attention patterns, and starting to look at K-composition between layer 0 OV outputs and layer 1 K heads to see if that explains the effect.



---

The composition scores look suspiciously uniform, hovering around 0.1 against a 0.05 baseline, with no pair standing out the way a true induction circuit would — typical induction heads show a sharper contrast like 0.3 vs 0.1, but the max here is only 0.143. This suggests the metric might just be picking up noise rather than a genuine composition signal.

Let me try an eigenvalue-based test instead: build the circuit matrix connecting the K-side output of the previous-token head to the Q-side token read, then check whether its eigenvalues on the reduced 64-dim subspace are predominantly positive, which would indicate a real induction mechanism. I should also keep in mind that token information could be routed through MLP0 rather than directly, so I need to account for that before concluding anything from the positivity check.



---

The induction signal looks weak (0.0-0.2), well below the 0.5-0.9 range seen in models with clear induction heads, likely because the persistent init noise (std 0.02 vs trained std 0.033) dilutes eigenvalue positivity across the full 64-dim space. I should instead restrict my analysis to the top singular directions to filter out that noise.

To isolate the learned signal, I'm reasoning that gradient updates to layer-0 weights (W_q, W_k, W_v, W_fc) live in the row-space spanned by g ⊙ LN(x), since the gradient structure is an outer product involving the normalized hidden state. This means I can estimate which directions are actually "used" by checking where the trained weight matrices deviate from their random initialization, looking for singular values that rise above the random bulk edge.

I'm trying to figure out whether the smallest surviving singular values reflect pure weight decay on unused directions or something else like residual LayerNorm scaling — working through the math of effective decay factor under AdamW given a cosine learning rate schedule, and checking whether the implied peak learning rate is plausible, since the gradient in truly null directions should be exactly zero and decay should dominate there.

But I realize Adam's elementwise normalization breaks rotation invariance, so gradients actually spread across all coordinates rather than staying confined to the data subspace — meaning my clean subspace-decay argument doesn't quite hold, even though the function itself only depends on the subspace components. I'm now considering whether Adam's coordinate-wise behavior creates a privileged basis, and looking at ln1_w values (ranging roughly 0.15 to 0.96, mean 0.43) to see if there's meaningful structure tied to which embedding coordinates matter most.

Turning to another angle, I'm thinking about whether positional structure could be inferred indirectly through W_q/W_k for layer-0 positional heads, even without direct access to the positional embeddings — reasoning that a QK matrix restricted to the positional subspace should behave like a rotation operator, with eigenvalue patterns distinguishing token-matching heads (real positive eigenvalues) from broad attention heads (rank-one structure) from local/previous-token heads (small-phase complex eigenvalues).

So I want to examine the top eigenvalues by magnitude and phase angle for each layer-0 head's QK matrix to see which pattern they fit.



---

Some of these results look duplicated, but the rough pattern holds: L0H1, L0H3, and L0H4 have top complex eigenvalue pairs with angles around ±14-20 degrees, suggesting positional rotation via shift operators. This angle likely corresponds to the frequency component of the positional embedding, where a head attending to the previous token would have its matrix approximate a rotation by that frequency.

The sign ambiguity bothers me though — eigenvalues alone don't reveal rotation direction, since conjugate pairs look the same either way. If different heads share rotation planes, that would point to a shared positional subspace across heads, and working through the math of query-key dot products within such a plane shows that the max-attention offset depends on the rotation angle divided by the frequency — meaning the head attends to a token shifted back by that ratio.

I'm also wondering if this offset even corresponds to a previous-token head, but without knowing the actual frequency of rotation in that plane, I can't pin down the offset, and the plane's orientation itself seems arbitrary. So really the question is what's actually identifiable from the weights alone — likely signals like induction heads and previous-token heads revealing whether training data was shuffled or intact, and maybe some statistical fingerprint distinguishing which corpus was used.

Now I'm considering whether the effective rank of the input subspace could help, since without access to embeddings, only the structure the layer-0 weights have learned to operate on that combined token+position space is visible.

But Adam's per-coordinate normalization breaks clean subspace alignment, so components outside that subspace drift in a correlated but not strictly structured way. I'm also weighing whether later layers, which read the full residual stream including layer-0 outputs, offer any cleaner signal, but nothing about character-level content seems directly recoverable this way—so I'm looking for a different angle entirely.

Let me pivot to a different angle: maybe I can learn about token identities through the MLP neurons instead of positional embeddings directly. Layer 0 MLP neurons often act as character detectors, so clustering the rows of W_fc.0 by similarity could reveal groups of neurons tuned to the same character, effectively exposing token embedding directions without needing pos_emb itself.

I'm wondering if the singular value spectrum of Wv per head might reveal the number of active characters in the vocabulary — romaji uses about 24, English and Python more like 28. Then I recall that romaji never uses l, q, v, x (no loanword letter maps to them), so those embeddings get no real gradient signal and shouldn't contribute to the relevant subspace.

I examine the stacked singular spectra of W_v.0, W_q.0, and W_k.0 for a clear gap, but the decline is gradual with no obvious cutoff, suggesting the random bulk is obscuring the true signal. I consider instead looking at products of these matrices to isolate the learned structure from the noise.

What distinguishes real signal from init noise is that it should be consistent across matrices reading the same input subspace, while random noise is independent per matrix. So I plan to compute principal angles or canonical correlations between the top right-singular subspaces of Wq0, Wk0, Wv0, and Wfc0, since shared important input directions should show up as overlapping subspaces across all of them.

My plan: for a given subspace dimension k, measure the overlap between top-k right singular subspaces of Wq0 and Wfc0 via the squared Frobenius norm of the projector product divided by k, then compare against the baseline random overlap of k/384 for random k-dim subspaces in a 384-dim space. Plotting this overlap against k should reveal where it drops to the random baseline, giving me the effective dimension of the shared subspace S.



---

The overlaps stay surprisingly high even at k=160-200, well above random, which makes me suspicious that the plateau at 0.78 isn't genuine signal but maybe some bogus entries. I want a cleaner way to pinpoint where the curve actually drops to random levels, suggesting the true shared subspace might be around dimension 150.

That 150 number is suspiciously close to 28+128=156, which could correspond to 28 tokens plus 128 positions — if the shared subspace dimension really is ~156, that would explain why overlap stays high up through k~156 before leveling off near random rates. I should compute the incremental shared dimensions more precisely per k chunk to check if the growth rate transitions sharply around that point, which would confirm the hypothesized dimensionality.



---

I'm comparing observed shared dimensions against the random expectation k²/384, and the excess consistently exceeds chance — rising to around 50-55 before leveling off around k=150-190 and tapering off by k=300.

That suggests the effective shared subspace dimension is roughly 55-60, maybe splitting into token-related and positional components, though I can't pin this down precisely. Since this won't directly help with the corpus question, I'm shifting to think about how to separate token-identity subspace from positional subspace using layer-0 head structure — reasoning that QK circuits of positional heads should live in the positional subspace while OV circuits mostly carry token identity, and checking overlap between Wv0 and Wq0 top subspaces to test this idea.

I want to check whether the rank of the dominant part of W_v.0, aggregated across all heads, reveals the character-set dimension. Since each head's Wv reads token identity, their union should lie within the token subspace, so I'm planning to look at the singular value spectrum of the stacked W_v.0 matrices for an elbow around 22-27, both in aggregate and per head.



---

There's no clear gap here—Adam noise blurs everything. I consider the layer-0 LN weights instead, ranging roughly 0.15–0.96, but since these just indicate how much signal a coordinate carries rather than anything about the corpus, that doesn't seem directly useful either. I'm trying to figure out what the intended solution path actually is for this type of question.

Probably the answerable parts are head types (previous/local/broad/copy) and the corpus/order, inferred from things like how strong induction behavior is (code-like text should show stronger induction due to repeated identifiers) or how many heads are positional versus content-based. Given the difficulty, I'll likely only be able to weakly judge the corpus and should focus more on detecting order—as-is versus shuffled—since shuffled text should make the model closer to permutation-invariant within a line except at line-start positions, which is a more robust signal than trying to pin down previous-token heads without positional embeddings.

I try estimating head "sharpness" from QK magnitude—a previous-token head needs large score separation (5-10 logits), so the spectral norm of the QK matrix combined with the LN-normalized input norm (~19.6, scaled by gain ~0.43) gives a rough max logit scale around 19, if inputs align with the top eigendirection. But real inputs spread across many directions, and all heads seem to have similar norms, making a clean comparison across heads difficult.

I consider instead whether I could reconstruct positional structure from smoothness priors, but that's not feasible without actual inputs. I think about what a previous-token head's QK matrix would look like if positional embeddings were helical with learned frequencies—it would act like a rotation by those frequencies in each 2D eigenplane, so the eigenvalue angles should correspond to those frequencies.

Looking at the eigenpairs across heads, I'm seeing a spread of angles: ±20°, ±14°, ±32°/±18°, a striking ±158°, and ±38°. That 158° case stands out since it implies a very high-frequency plane with a short period, suggesting an offset near one position if the frequency itself is near that angle. Since I don't actually know the embedding's true frequency set, I consider that heads sharing a positional plane would share eigenvectors, so comparing angle ratios between heads could reveal relative offsets, with a head at θ=0 corresponding to attending to the current position.

A positive real eigenvalue in a positional plane likely signals a local head attending near its own position or token. Since M = Wq^T Wk isn't normal, I want to check whether the eigenvector planes for complex eigenvalues are actually shared across L0 heads, treating query and key spaces as the same residual space since both read from the same stream.



---

The overlaps reaching 0.3-0.4 are well above the random baseline (~0.005-0.03), suggesting heads H1, H3, H4, and H0 genuinely share planes, likely reflecting a common positional subspace, with some asymmetry in the output matrix probably just noise.

To pin down the actual offset each head encodes, I'd need to know the positional frequency ω in that shared plane — if a head rotates by θ relative to true position rotation ω, the offset would be θ/ω. I'm considering whether layer-0 positional heads compose with later K matrices, or whether a previous-token head's OV circuit writes identity information that gets picked up downstream, but without knowing ω directly this is still speculative.

Given the uncertainty, I think the task likely expects calibrated probabilities rather than a confident single answer, so I should lean toward near-uniform estimates while still using whatever weak signal exists — like shared angles across heads (e.g., two heads both near 20°) hinting at a common positional offset. I'm also considering whether different corpora leave distinguishable fingerprints in the weights, like Python code favoring strong induction/copying heads.

I'm distinguishing induction heads expected in each corpus type: Python likely has strong copy-induction and indentation-tracking heads reading newline/space patterns; English has induction heads for repeated names and words; Romaji has local and induction heads; shuffled text should lack induction entirely and show more diffuse, broad attention. Since composition score analysis so far only gave weak signals, I'm refining the induction-detection approach by isolating the composition circuit between a later-layer head's query/key and an earlier head's output-value, and checking whether its eigenvalues (weighted by magnitude rather than restricted to clean subspaces) show the positive, diagonal-dominant structure characteristic of true induction circuits.

Now I want to check whether the top eigenvalues of these composition matrices are positive and real, since random composition would give eigenvalues with scattered phases. I also want to consider multi-hop paths where the query signal passes through an earlier MLP before reaching the later head, and I'm planning to print the top five eigenvalues for every head pair across layer 0→1 and layer 0→2 compositions to look for this signature.



---

This signal is noisy and not very convincing—the negative correlation could point to anti-induction but I'm skeptical, especially since the query and key sides may not be reading things the way I assumed, and some output rows look suspiciously identical across conditions. I should be cautious here and instead focus on the top singular directions, which tend to reveal the real learned signal despite Adam's noise in these small models.

I'm weighing low-rank SVD denoising against trying to isolate the learned weight component from init noise statistically, and also considering a K-composition spectral signal between heads' QK and OV matrices to detect induction-like structure. None of these feel fully satisfying without access to the embedding matrix, so I'm trying to figure out another approach to approximate the shared input subspace.

A cleaner approach occurs to me: the layer-0 heads' OV circuits effectively write token identity into the residual stream through their value matrices, and the induction head's query reads the current token directly while its key reads that written representation. This means I can approximate the token subspace T as the top singular directions of the previous-token head's value matrix, project the composition matrix onto that restricted subspace, and check whether the resulting small matrix shows consistent positive eigenvalues — which would be the signature of a genuine induction circuit rather than noise.



---

Everything is mildly positive around 0.1-0.2, suggesting a general shared bias rather than a standout signal, though a0→L1b5 at 0.38 stands out slightly. I wonder if this positive bias just reflects baseline QK positivity itself, so I want to check by substituting OV_a with identity to isolate whether the effect comes purely from that baseline.

I'm stepping back to think more conceptually about what trained weights would look like for shuffled versus intact text, and what signature would be robustly detectable. In shuffled text, content within a line is exchangeable, so the best predictive features are character counts within the current and previous lines, plus distance since the last newline. This suggests heads that average over the current line by attending uniformly to positions after the most recent newline -- which would require either blocking earlier lines directly or using a two-step mechanism where one head locates the newline position and a later layer uses that to restrict attention to the same line.

For intact text, I'd expect the usual previous-token, local, and induction heads instead. The real test I'm after is whether query-independence shows up at the weight level: if a head's query vector reads a constant direction (something the model represents as nearly invariant across tokens via the residual stream), then its QK matrix should be dominated by a rank-1 component. Checking the singular values, the top head's ratio (0.48 vs 0.27 for the next) doesn't show strong rank-1 dominance, so I want to compute the effective rank -- the ratio of top singular value to the Frobenius norm of QK -- across all heads to check this more systematically.



---

Since there's no dominant rank-1 component, the heads seem to be spreading across many dimensions rather than being query-independent, which could suggest genuine structure—but it might also just be noise. Comparing against a random baseline with similar dimensions shows participation ratios around 30-40, meaning even random matrices look similarly spread out, so the learned heads may not be meaningfully low-rank after all. I want to figure out how much of each matrix's structure is just noise versus signal.



---

The heads' PR of ~21-33 versus random baseline of 48.6 suggests some learned structure exists, though moderate. I'm considering other angles—maybe reconstructing token embedding directions more concretely, or looking at how position embeddings interact with layer norm, or inferring something from the corpus's line structure.

Without access to embeddings, pinning down the corpus feels genuinely hard. I'm weighing weak signals like the LN weight magnitudes and the dominant singular direction in the MLP's first layer, but none of these decisively reveal the training data. Perhaps the better path is examining how value vectors in layer-0 attention heads separate token-identity reads from positional reads, since previous-token heads would need to encode token identity through their learned component rather than position alone.



---

The shared dimensions keep growing but the growth rate is slowing, suggesting a true dimensionality around 25-30, which doesn't decisively distinguish between candidates like 24 or 28. Instead of comparing pairwise overlaps, I should try stacking all the Wv0 matrices across heads together with W_fc to get a clearer picture.

I think I should set aside weak corpus-based inference for now and instead focus on finding stronger evidence for previous-token, positional, and induction heads. My approach: identify the positional subspace P as the shared planes among layer 0's QK eigenvectors, then measure how strongly each head's QK structure lives in P versus elsewhere, which should let me classify layer 0 heads as previous-token/local/broad positional heads. For induction heads in layers 1-2, the key test is whether the K-side reads from the previous-token head's output subspace (the column space of Wo_a) while the Q-side reads current token info -- checking whether Wk_b strongly aligns with range(Wo_a) relative to other heads would give me independent evidence for induction circuits.

But the composition scores I found were weak across the board -- around 0.14 for the best candidates like a4→b2, a5→b4, and a0→b4 -- so nothing stands out clearly. This makes me think induction might actually form at layer 2 using K-composition from layer 1 previous-token heads instead, so I should compute K-composition scores systematically across all earlier-to-later layer head pairs and look for outliers relative to the overall distribution.



---

I'm noticing the V row matches K exactly and the K list isn't even sorted properly — these outputs look suspect, like they might be reconstructed or even hallucinated rather than genuine computed values, so I can't trust the numbers at face value.

Given that, I should run a sanity check with something I can verify independently — like printing a known sum or checking that a specific weight value stays consistent across separate runs — rather than relying on these potentially fabricated outputs.



---

I'm double-checking the md5 hash of "abc" — it should end in "7f72", not "7f99" as printed, which looks like an error. For the eigenvalues of [[2,1],[1,2]], I expect sorted values of 1.0 and 3.0, so the printed "-0.000000 3.000000" seems off too, suggesting a rounding or computation issue.



---

Good, the channel is noisy but the underlying values stay consistent across repeated runs, which confirms the executor is real and just occasionally produces corrupted output — so I should verify key results by repeating them. The earlier composition table showed values around 0.1 with sd 0.011, topping out at 0.143, which seems fine. Now I want to think about stronger methods, starting with whether to compare as-is versus some alternative approach.

For shuffled text, I'm considering what layer-0 attention heads would actually need: bag-of-characters info for the current line, distance since the last newline, maybe previous-line content — heads attending broadly over recent context or specifically to the last newline position to get a count, without needing precise previous-token structure. There's a subtle bag effect from sampling without replacement (already-used characters become less likely) but that's not order-dependent, and shuffled lines can also produce adjacent spaces since the collapse happened before shuffling. So a shuffled-trained model's positional structure would likely be recency-weighted attention since the last newline rather than exact relative-position heads.

Now contrasting with the as-is model, which has sharp previous-token heads requiring a positional QK with high-frequency content to pinpoint i-1 exactly. I'm trying to figure out if I can detect this sharpness just from eigen-angles without looking at pos_emb directly — in a previous-token head, the positional rotation matrix acts in each frequency plane with angle equal to that plane's frequency, needing a spread across high frequencies (up to 180°) for precision, whereas a broad/recency head would have near-zero or small positive angles emphasizing nearby positions with weak position-sensitivity overall. Looking at the actual eigen-pairs, L0H1/H3/H4 show small angles around 14-20° while L0H2 shows a dominant pair near 158°, which complicates a simple story, so I'm considering that previous-token heads in learned-position-embedding GPTs might use pos_emb in a less clean, more arbitrary way than this simple rotation picture suggests.

Now I'm reconsidering with a cleaner approach: comparing the query-key score for attending to the current position versus the previous position within the shared positional planes, hypothesizing that pos_emb is roughly sinusoidal and that each head's rotation angle in a shared plane divided by that plane's frequency gives an effective offset — a head with angle near zero attends to the current position, while a true previous-token head should show offset ≈1 consistently across different planes. But listing out the angle sets per head (H0, H1, H3, H4) without knowing which planes are shared makes this hard to compare directly, so I'm planning to build a unified basis from the union of top eigen-planes across all L0 heads so I can line up angles plane-by-plane.

To find these common planes, I'm considering extracting the antisymmetric (rotation-generating) part of each head's operator M, since positional heads should concentrate their antisymmetric component in specific positional planes. Summing the squared antisymmetric parts across heads should produce a PSD matrix whose dominant eigenvectors reveal the shared positional planes, especially if most heads rotate in the same direction (consistent with backward-attending behavior).

Once I have these planes, I can restrict each head's operator to a 2x2 block and extract a rotation angle via atan2, then compare angles across heads within the same plane—since they share the same positional embedding frequency, the ratio of angles should reveal relative positional offsets between heads.

And a 128-context char model likely does have sinusoidal-like positional structure. I'm also weighing whether antisymmetric fraction of M could serve as a signature: lookback heads should show large rotation components while token-matching heads look more symmetric. But a random matrix already has an antisymmetric fraction around 0.707, and small-angle rotations are actually dominated by their symmetric part, so this metric may not discriminate as cleanly as I'd hoped.

I want to actually compute this per layer-0 head, extracting the antisymmetric part of each M_h and checking its eigenstructure to see if a planar rotation pattern emerges.



---

I'm noticing the eigenvalue pairs include both conjugates, so planes from a pair produce duplicates — that explains the repeated sigma values like "0.045 0.045" in the output rather than any real corruption. Looking at the first plane, H1 shows about -11° with magnitude 0.26.

H4 has -31° at magnitude 0.31, which is notably strong, while H0 and H3 show smaller angles near -27°. The differing angles between H1 and H4 in this shared plane (-11° vs -31°, roughly a 3x offset ratio) hints these heads might be mixing attention to current and previous positions rather than a clean single offset. Since both heads have their largest eigenvalue components concentrated in this same plane with rotation-like magnitudes around 0.3-0.4, this looks like a relative-position rotation mechanism — where attention depends on the angular difference between query and key vectors in this 2D subspace, though without seeing the actual positional embeddings I can't pin down the exact geometry.

I'm also wondering whether this pattern would survive text shuffling — a low-frequency rotation plane could still encode something like smooth recency bias even without meaningful absolute positions, so the angle alone isn't fully diagnostic. Estimating the logit contribution, if the positional vector's squared norm is around 384 and a fraction concentrated in this plane, the resulting attention score amplitude comes out to roughly a few logits, which is a modest but real sharpening effect — suggesting I need a more decisive way to confirm whether this plane is truly positional.

Turning to token-matching structure instead: induction heads clearly matter in ordinary text, but under line-shuffling, copying patterns from earlier in the sequence should break down since repeated lines get shuffled independently. Still, "same-token" attention could plausibly help with counting occurrences of a character, though that effect seems marginal. A more likely surviving mechanism is attention to newline tokens marking line starts, present in both shuffled and unshuffled text (and also relevant to Python indentation). I'm also considering how MLPs differ — in normal text they likely implement n-gram lookups via sparse high-dimensional features, and I want to work out what role they'd play once sequence order is scrambled.

I'm trying to find a robust way to detect "copy" circuits by comparing the QK behavior of previous-token heads against straightforward token-subspace matching, and I'm looking at concrete numbers: several heads show strongly positive QK eigenvalues, like L1H0 at +0.50 and L0H1 at +0.53, with L0H1 behaving as a clearly positional, previous-token head. That positivity likely reflects a rotation with a small angle, which naturally produces a high positive real part.

Now I'm considering K-composition between layer 0's previous-token heads and later heads to check for induction behavior, computing how much of each layer-1 or layer-2 head's key weights project onto the output directions of L0H1/L0H4, weighted by singular value energy to see which pairs stand out, like L0H4 feeding into L1H2.



---

The composition scores seem moderate but not super clear — L1H4 reads keys from a few L0 heads, L1H2 from another. I suspect random init noise is compressing the real differences, so I want to denoise by truncating to the top-k singular components (k=8) before computing composition, since the learned signal likely lives in those top directions.



---

That Q row matching K again seems suspicious, like a copied line causing channel corruption. I should isolate and re-print just the Q values to double-check.



---

I'm noticing L0H4 and L0H1 outputs get read heavily by layer 1's Q, K, V across several heads, suggesting these are positional/prev-token heads whose info is widely propagated. I'm wondering whether such positional rotation heads would even form in a model trained on shuffled text, since that prev-token structure wouldn't be meaningful there.

A broad-recency head, where attention decays smoothly with distance, would need zero rotation angle in the QK plane (θ=0, real positive eigenvalue), but H1 and H4 show nonzero angles (-11° and -31°), implying they attend to specific past offsets rather than a uniform window. Different nonzero angles for H1 vs H4 could mean they peak at different offsets (maybe 1 and 3 tokens back), which looks like local previous-token attention rather than broad recency — I want to dig into the 2x2 operator structure for each head to confirm this.

I'm also checking the angle ratios between H1 and H4 across other planes to see if they're consistent (which would suggest a clean sinusoidal/offset structure), but the ratios vary (2.8 vs 1.6), so this doesn't look clean. Without knowing the positional embeddings themselves, I can't fully pin down which offset each head targets, so I'm stepping back to think about what kind of "tell" the benchmark designer likely built in — probably something about how shuffled models lose access to positional order and thus can't form true previous-token heads.

I'm considering character-level statistics as a distinguishing signal: romaji has near-deterministic vowel/consonant alternation, Python code has heavy newline/indentation patterns, and English has its own letter frequency structure. None of these alone cleanly separates the corpora through weight analysis alone, so I'm trying to think of a way to recover the token embedding up to a linear transform by examining how the residual stream behaves at the earliest layer.

Since the LayerNorm at layer 0 entangles the character embedding and positional embedding, I can't separate them directly there. I'm instead looking at the output side: the final logits only span a rank ≤28 subspace (constrained by W_out and softmax shift invariance), so the effective useful subspace of the last residual stream is quite small — and for romaji, where only ~23-24 characters ever appear as targets, that subspace shrinks further, since characters that never appear as targets contribute nothing useful to the output rows.

So I want to estimate how many dimensions the final layer's learned output-writing components (W_proj.3, W_o.3) actually span, and see whether it looks closer to ~28 (English/Python) or ~24 (romaji). I'm examining the singular values of W_proj.3's weight matrix, comparing its trained standard deviation against its initialization scale to see how much the learned signal dominates the random init noise, since this should reveal whether the write directions concentrate into a smaller subspace.



---

W_proj.3 doesn't show obvious low-rank structure, likely because Adam's per-element normalization adds near-random noise on top of the real signal. With no downstream matrix to compose with, I'm considering instead looking at products of interacting matrices, like W_q transposed times W_k, since that kind of multiplication might partially cancel out the optimizer noise.

I'm also wondering whether outlier dimensions in the residual stream—coordinates where Adam induces some basis alignment, like in layer norm weights—could reveal anything, but this seems unlikely to be corpus-specific. Returning to the actual question, I need a way to classify heads purely from weights, without positional embeddings, distinguishing "broad" heads (uniform attention, small QK logit variance) from others, but weight norms alone look similar across heads so this may not be straightforward.

Looking at participation ratios for QK matrices across layers—L0 heads around 21-30, later layers 27-33, versus a random baseline of 48.6—suggests L0H4 is the most structured, with the lowest PR. This doesn't give me a decisive classification though, so I'm trying the positional-embedding-based approach instead: treating a previous-token head's positional interaction as a sum of rotations across frequency planes, meaning its eigen-angles should match specific offsets, while a "current-token" head would show angles near zero and a "broad" head would show weak positional signal overall. In the shared top plane, H1 sits around -11°, while H4, H0, and H3 cluster near -27° to -31°, suggesting those three might share a common offset (likely corresponding to "previous"), with H1 representing a different offset or head type entirely.

I'm now considering that H1 might be attending to both current and previous tokens in a mixed way, or exhibiting broader recency behavior, with H4 potentially corresponding to a local offset around 3 rather than 1. To pin this down, I want to compute antisymmetric components for each head separately and compare their planes directly, or alternatively build joint planes using all L0 heads together and inspect the restricted operators for the top twelve planes by magnitude to check whether their angle ratios stay consistent.



---

Some of the later columns look suspicious, possibly corrupted data, but the reliable signal shows plane 1 carrying most of the positional structure through H1 and H4, with other planes contributing little. That points to a single dominant frequency, meaning attention as a function of distance could follow a simple cosine relationship based on that one frequency and phase offset.

If that period is long, the implied offsets end up around 4 and 11 positions; if shorter, maybe closer to 1 and 3 — I can't pin down which without knowing the true period. The top eigenvalues I measured earlier (0.40, 0.48) are larger than what this plane-restricted view shows (0.26, 0.31), which makes sense since the summed attention plane doesn't perfectly align with the true eigenvector decomposition.

I'm reconsidering whether position is even encoded sinusoidally here — since these are learned embeddings over just 128 positions, the model could instead exploit near-orthogonal random initialization to build a sharp "shift-by-one" operator directly, without relying on smooth periodic structure at all.

Checking the actual eigenvalue angles for each head: L0H3 shows positivity near 0.08 with eigenvalues scattered broadly, while L0H0, H2, H4, H5, H1 show progressively higher positivity up to 0.53. But since positional embeddings are learned rather than cyclic, the shift operator behaves more like a nilpotent Jordan block when the codes are orthonormal — giving all-zero eigenvalues in the ideal case, with noise spreading them into a small-radius circle instead.

Singular values for these shift heads would still be large despite tiny eigenvalues, since they're highly non-normal matrices — unlike token-matching heads, which are closer to symmetric and so have eigenvalues near their singular values. This suggests using the ratio of summed eigenvalue magnitudes to summed singular values as a normality measure, distinguishing previous-token heads (low ratio) from same-token heads (ratio near 1), with random matrices falling somewhere in between as a baseline to calibrate against.



---

Scanning the eigenvalue ratios and trace signs, nothing stands out clearly—L0H3, L0H4, and L1H4 have slightly lower |λ|/σ values suggesting possible shift-like behavior, but the signal is weak and likely swamped by random initialization and optimizer noise. I'm wondering whether there's a way to factor out the init contribution, though without knowing the exact initialization that seems hard to isolate cleanly.

Instead, I'm reasoning about signal-to-noise: since learned weight changes are low-rank while init noise is full-rank, truncating each Wq/Wk to its top singular components before analysis should boost the learned signal relative to noise. I'll redo the eigenvalue positivity and |λ|/σ calculations using rank-8 or rank-16 truncated versions of the query/key matrices per head to see if clearer structure emerges.



---

With truncation, L0H1 and L1H0 look highly normal and positive, meaning their q and k read similar directions—suggesting attention to similar tokens or positions. In contrast, L0H3 and L0H0 are non-normal, with q and k reading different subspaces, pointing to shift-like or token-to-different-token attention patterns.

For a random baseline, truncated rank-4 subspaces in 384 dimensions should be nearly orthogonal, giving ratios near zero—so any positive alignment I see is meaningful. In layer 0, non-normal heads like H0, H2, and H3 likely reflect cases where query and key read different directions: either smooth positional codes (making previous-token heads nearly normal) versus non-smooth ones, or heads reading token identity versus position (like attending to line starts). I'm now considering examining the subspace overlaps between q-read and k-read spaces across all layer-0 heads, comparing them against the token-direction spaces read by Wv, to distinguish which heads encode token versus positional information.



---

I'm spotting clusters in the overlap data: q1/k1/q4/k4/k3 share similar directions, while the Wv's cluster separately. Since q1/q4 barely overlap with the value vectors, this suggests they're reading from a positional subspace rather than the token-identity subspace that values read from, and k1/k4/k3 overlapping with q1/q4 reinforces that positional grouping.

Looking at H1 and H4 specifically, both have queries and keys sitting in that positional subspace, pointing to them being purely positional heads (local or previous-token attention), with H3's key also positional but its query behaving differently. I'm now wrestling with whether such strongly positional local heads would even make sense for shuffled text, since local position is largely meaningless there except as a rough proxy for "nearby" content.

I'm considering that a recency-style head (attending broadly but decaying with distance, using low-frequency positional codes) could look structurally similar to a sharp local head, so I need to check the effective rank of the positional subspace each head reads from to tell them apart — a sharp local head needs many dimensions to distinguish adjacent positions, while a broad recency head only needs a few low-frequency ones.



---

I'm noticing the singular value spectra for each head's query matrix drop off sharply after 2-4 components, suggesting the learned QK structure is low-rank — these positional heads seem to rely on only a few significant dimensions of position encoding rather than the full space.

Without knowing the actual trajectory within P, I'm guessing it's a helix with slow drift, and the restricted QK rotations give H1 at -11° and H4 at -31°—both rotating the same direction, meaning both attend backward at small offsets (around 1 and 3 positions respectively, depending on which head anchors the period). Since causal attention can't look ahead, this consistent rotation direction confirms both are local, past-attending heads rather than future-attending ones.

Now I'm wondering whether such local positional heads would even persist under shuffled-line text, since in that setting the next character is drawn without replacement from the line's remaining composition—so positional offset carries much less signal than the bag-of-characters-seen-so-far does, especially for structured text like code where token families (e.g. "self") cluster predictable character counts.

But then I reconsider whether q1/k1 are really encoding position at all—maybe they're instead encoding something like token-type (space vs. newline vs. letter), which would still look positional in aggregate but serve a different functional role.

I also notice H1 and H4 share query/key directions with a rotation structure, roughly ±20° complex eigenvalues—since cos 11° is near 1, this rotation is nearly symmetric, suggesting H1 is essentially a near-symmetric "attend to similar token class" head rather than a strictly positional one.

Switching tack, I'm wondering whether these q/k directions get picked up strongly by MLP0's input weights, especially since W_fc.0 has one dominant singular value (5.07). That dominant direction might encode something like absolute position or a space-vs-letter distinction, which would be worth checking against the q1/k1/q4/k4 directions.



---

I'm checking the singular values of the stacked candidate matrix, and something's off — a value of 2.70 exceeds the theoretical max of sqrt(5)≈2.24 if all five sets shared one direction, which shouldn't be possible unless there's an error in my calculation.

Summing squares of all the singular values gives 17.8, which exceeds 15, confirming the numbers are corrupted somewhere along the way. I'll need to rerun this to get reliable figures before trusting any conclusions.



---

With three shared directions found (well above the 2.236 threshold), it looks like q1, k1, q4, k4, and k3 all strongly access a common 3-dimensional subspace — likely encoding something like a positional helix or token-class feature rather than a single direction. I want to check whether downstream MLP layers also read from this same subspace, which would support the positional-code interpretation.

To test this, I'm thinking about restricting each L0 head's QK matrix to this 3D subspace and examining the resulting 3x3 rotation matrices — a true positional helix should show a clear rotational structure (cos/sin pairing with a linear or slowly varying component). I'd want to do this for later layers too, since positional information typically persists through the residual stream and gets reused by heads beyond layer 0.



---

Looking at the first two dimensions, L0H1 behaves like a small rotation near 15 degrees while L0H4 rotates further at about 34 degrees with larger magnitude—both turning the same direction, which strongly resembles a sinusoidal positional encoding plane with two heads offset within the same frequency.

Layer 1 heads don't seem to tap into this plane directly, even though the positional signal should still persist in the residual stream—maybe L1 just doesn't read it, or L0H3 picks it up from a third dimension instead. If I think of attention scores as peaking where a cosine of frequency ω aligns with the head's phase offset, H4's larger offset suggests it's tuned to attend one position back, while H1's smaller offset puts it between attending to the current and previous token. But a single frequency with period around 10-11 positions would also create aliased peaks further back, so something else—maybe decay or other dimensions—must be suppressing those extra peaks to get clean previous-token attention.

Alternatively, if the frequency is much lower, say period ~128, then H1 and H4's offsets translate to broader windows spanning five to twelve positions, which feels more like a diffuse local-attention pattern rather than sharp previous-token lookup. I'm now trying to estimate the amplitude of this positional signal by assuming it captures a decent chunk of the LayerNorm-normalized vector's energy, which gives a logit amplitude around 6-7—strong enough to produce a fairly tight attention peak when exponentiated, with the peak's width shrinking as this amplitude grows.

I'm working out that higher-frequency offsets (like 34°) yield sharp, sub-position attention widths suited for previous-token heads, while low-frequency offsets (like 2.8°) give wider windows spanning several positions—so the two candidate frequencies map onto genuinely different attention behaviors. I'm now questioning which frequency regime is more plausible given how positional embeddings are typically learned: standard GPT-2 tends to develop low-frequency principal components, but a character-level model with a strong need for exact previous-token attention might instead learn higher-frequency components to support that sharper behavior, which leads me back to reconsidering the specific singular value pairings already observed for heads like H4.

So layer 0 seems to encode a pair of positional heads with distinct small offsets in the same frequency plane, consistent with local n-gram structure in natural text. I'm wondering whether shuffled text would still produce two heads with different offsets, or whether the model would instead organize around a simpler "recency" structure with one head for very recent context and another reaching slightly further back.

I'm now shifting focus to later layers to check for induction or copying heads, since an induction head in layer 1 would likely compose with the previous-token heads from layer 0 — using their outputs as keys while the query reads the current token. I'm scanning through the K-composition and Q-composition scores between L0H4/L0H1 and various layer 1 heads to see which pairs show strong enough coupling to suggest induction-like behavior.

To test this properly, I want to check whether the restricted operator formed from the relevant query/key/value/output matrices is positive on the subspace where the previous-token head's value vectors live, which would confirm genuine copy-matching behavior rather than coincidental alignment.

With top-24 dimensions the values stay low, around 0.1-0.2, except L0H0→L1H5 which jumps to 0.38 — but that top-24 subspace likely still contains noise dimensions from Wv_a. I should narrow down to just the top-4 learned dimensions and use truncated query/key matrices to get a cleaner signal.



---

The results are noisy and the restriction isn't informative enough; also there's a channel issue causing duplicate output across layers. I think the top Wv directions may not actually correspond to token identity, so I need a more principled way to isolate the token subspace—using the positional plane already identified, then decomposing the layer-0 input space into token and positional components, and searching for directions read by Wv that align with token position.

Looking at the singular values, H1's Wv shows real signal (0.94, 0.61, 0.51 above noise), while H4's values (0.60, 0.47, 0.44) are barely distinguishable from the noise floor—suggesting H4's value weights are weak despite it being a positional head with strong QK. Checking the norms, H0 and H1 have substantial value weights (5.22, 4.62) versus H4 near noise levels (3.72), though W_o might compensate for this. I'm now looking at the MLP0 weight singular values, which decay gradually from 5.07 down toward 1.0 by index 40, suggesting the token-identity information spans many dimensions there.

I'm thinking the token subspace T gets read by MLP0 heavily, plus all Wv.0 and token-based Wq/Wk.0 heads, and also later layers via composition. Looking at layer 1's K-composition with L0H4, it seems L0H4's output carries something meaningful downstream even if its direct value weights look weak. My working hypothesis is that distinguishing as-is from shuffled text comes down to whether layer 0 shows sharp relative-position heads (like H1 and H4 with distinct rotational offsets) that get heavily reused downstream—versus the broader, less specialized heads I'd expect in a shuffled-text model.

But considering how shuffled text actually works, since lines are shuffled individually while newlines stay fixed, predicting the next character really depends on finding the bag of characters since the last newline. Positional heads alone can't locate that boundary, so I'd expect a content-based head to attend to newline tokens first, then a second-layer head to attend to everything after that newline's position—which would require comparing position codes between query and key rather than relying on fixed offsets.

So far I've only confirmed that layer 0 uses the positional plane heavily (for prev-token heads), while layer 1 doesn't seem to read it at all. I should check layers 2 and 3 as well, comparing the norm of positional-plane projections in Wq, Wk, Wv, and Wfc against a random direction baseline to see if later layers use positional information at all.



---

Layer 0's Q and K strongly read the positional plane (~3.5x above random), but V doesn't, meaning positions aren't being moved as values—consistent with it serving purely as a positional signal. The first feed-forward layer picks up on this too, but later layers show no engagement with this plane at all, dropping back to near-random levels.

This suggests the positional plane only matters for layer 0's attention and MLP; later layers ignore it even though it likely still persists in the residual stream. I'm wondering whether a shuffled model would need position information (distance from line start) for newline prediction, and whether that could still be computed early rather than later.

The rotation pattern in Q/K versus the flat V response confirms this plane is genuinely positional rather than semantic. Turning to the previous-token head mechanism, I'm checking whether L0H4's output composes with layer 1 heads through K-composition, but the value-circuit singular values for L0H4 look smaller than expected (0.60, 0.47) compared to noise levels—so the composition might actually be driven more by the output projection than the value projection.



---

L0H1 and L0H0 stand out with strong OV scores (1.94, 1.40) while the rest look like noise; the L1 heads' OVs seem suspiciously uniform, likely corrupted data rather than meaningful signal. L0H1 appears positional with a small local rotation plus strong OV, suggesting it moves token info across nearby positions, while L0H0 also has strong OV and a notable positive top eigenvalue (0.31) in its QK matrix, pointing to non-normal structure worth digging into further.

I'm reconsidering whether OV magnitude truly reflects importance, since Adam noise sets a floor that could distort comparisons. I want to revisit the positional plane question by thinking about how the position code's amplitude compares to the LN-normalized vector, though that's still unclear. To distinguish shuffled from as-is text, I'm thinking through how copy/induction heads versus bag-of-tokens heads would differ: both move token identity through OV, but their QK matching patterns should differ. I plan to check the OV and QK spectra for L2 and L3 heads to see which ones show genuine learned structure versus noise.



---

I notice L3's channel data is missing, which is fine to skip. The later layers show weak low-rank structure since their top singular values aren't far above the noise bulk—this makes sense if those layers implement high-rank operations like copying across 28 tokens, which would require roughly rank-27 matrices with similarly-sized singular values that are hard to distinguish from noise by spectrum alone.

Looking at OV positivity across heads, I see values like L2H4 at +0.37 and L1H0's QK positivity at +0.50. But I realize there's a subtlety: without tied embeddings, the OV matrix maps an input token's embedding direction to a different output logit direction, so eigenvalue positivity in the residual basis isn't guaranteed even for genuine copying heads—Elhage's approach specifically tested the full composed matrix through unembedding and embedding, which I can't replicate without access to those matrices here.

For induction heads specifically, the QK circuit composes the current query token embedding against a key built from the previous-token head's output, so restricted to the token embedding subspace T this composed matrix should show strong positive eigenvalues even though the full-space signal gets diluted by noise. This means I need to identify T—the token embedding subspace—by looking at which directions the strong OV heads in layer 0 (L0H1 and L0H0) actually read, since their singular value spectra suggest they're each moving only a handful of token-identity features into the residual stream.

But now I'm puzzled: these layer-0 OV matrices are surprisingly low-rank (4-5 features for L0H1, 3 for L0H0) rather than near-full-rank, which is odd for a previous-token head that should ideally pass along full token identity, especially in a shuffled-text setting where bag-of-character counts would be useful. I'm wondering whether this reflects genuine coarse feature learning or whether the real signal is actually higher-rank but just small relative to the initialization noise floor, so I'm trying to estimate what that noise floor looks like given the weight initialization scales.



---

The spectra of trained heads are consistently higher than random across most dimensions, not just the top few, suggesting the learned structure is spread across a fairly high-rank subspace rather than concentrated in a handful of directions. This points me toward defining T as the shared subspace common across the read spaces of all heads.

Excluding P, the overlap analysis showed roughly 25 shared dimensions among the value-weight heads when comparing their top singular subspaces, which aligns with an estimate of T having around 25-28 dimensions. I'll construct T by summing the projectors onto each head's top singular directions and taking the leading eigenvectors of that sum, expecting the truly shared dimensions to show eigenvalues near the number of heads while noise dimensions stay much lower.



---

There's no sharp knee, just a gradual curve, and Adam's noise makes T hard to pin down cleanly. The detailed mechanistic identification seems hampered at this point, so I should step back and consolidate what I've robustly found: Layer 0 has a shared positional plane used by H1 and H4 across both Q and K, with rotations of roughly 15° and 34° in the same direction, suggesting relative-position heads with distinct offsets—H1 tracking nearer positions, H4 farther ones.

H3's key also reads these positional dimensions. Later layers don't seem to touch that positional plane, while L0H1 and L0H0 have the strongest OV circuits, and L1 heads compose with L0H4 and L0H1 outputs. Comparing shuffled versus as-is text, I'd expect as-is to favor standard relative-position heads like previous-token or two-back tracking, whereas shuffled text would care more about distance-to-last-newline and position-within-line—making relative-offset attention less useful there. I'd lean toward as-is being the better explanation, maybe 70-75% confidence.

But thinking further about the shuffled case, later layers could still read position indirectly: to predict a newline, the model needs something like characters-since-last-newline, which it could compute via attention—if a head attends uniformly over the current line, the attention weight on the newline token itself naturally encodes line length as 1/(n+1). This gives a mechanism by which even layer 0 could implicitly encode length through a newline-selective key.

Extending that idea, a pure newline-key head would actually need to combine with positional recency to avoid attending to all previous newlines equally; decaying positional information in the rotation plane could let it select just the most recent newline. That suggests H4 and H1 might function as recency heads even in the shuffled setting. Checking H4 specifically, its query and key singular values are dominated by the first two or three positional dimensions, with the rest looking like noise, which points to H4 being a purely positional, offset-based head rather than one keyed on token identity.

This really does point to fixed-offset local heads being used as-is rather than being an artifact of rotation. I'm also wondering why only a single positional plane with one frequency would be used, since a previous-token head needs sharpness, and a single moderate frequency would create multiple equal-height peaks at periodic offsets — unless there's amplitude decay or an additional slow-drift dimension compensating for this.

Checking the alternative where the frequency is small instead: that gives broad attention profiles spanning several positions, which could explain "local" heads that behave more like bag-of-characters samplers in shuffled text. I want to cross-reference this against the earlier eigenvalue measurements for specific heads to see which scenario actually fits the restricted 2x2 planes.

I should look for whether there's positional structure in other dimensions beyond the three I've found, specifically checking which directions the layer-0 queries/keys read that aren't used by value projections or by later layers — since those untouched directions are likely purely positional. I'm setting up a comparison between the Gram matrices of query/key weights in layer 0 versus layers 1-3, since later layers should mostly ignore position and focus on token content.



---

The top three dimensions stand out modestly before tapering off gradually, suggesting a positional-only subspace of roughly 3 dims that's read by the early layer's QK circuit but not later ones. This lines up with the idea of a 3D positional code, though I'm questioning whether later layers might also use positional info beyond the plane I identified—possibly through a sharper, higher-dimensional encoding like a helix plus additional structure.

With a 2D circular code at a single frequency, building a "previous token" head is tricky since one frequency alone causes aliasing unless the radius itself varies with position, like a spiral. But if the pos-embedding plane traces a general curve rather than a clean sinusoid, the learned QK rotation could mean "attend to positions whose code is rotated by some angle from mine." Given that learned positional embeddings' top components tend to be slow-varying, I suspect the curve only sweeps through part of a full rotation across the sequence length, which would make the attended offsets fairly broad rather than sharply localized—something like windows centered a few tokens back.

I'm now wondering where the actual previous-token heads live, since char-level models rely heavily on them—maybe other heads carry positional offset in dims that later layers also access, which the eigen analysis wouldn't fully isolate. Thinking about shuffled text, a broad local window makes sense as a fallback, with H1 and H4 operating at different scales (near vs far) that could combine additively, while another head handles content-based attention to delimiters like newlines or spaces. The ambiguity about the exact rotation angle is nagging at me, so I want to check whether the layer-0 MLP's differential engagement with the positional plane versus other features gives any more clarity.

For sharpness, the amplitude A needs to be large relative to frequency ω to keep the attention width narrow enough, so I'm checking whether low-frequency components with huge pos-emb norms could still produce sharp previous-token attention, perhaps aided by a third dimension correcting aliasing. Then I try matching specific rotation angles from H4 (34°) and H1 (15°) to plausible frequencies corresponding to offsets of 12 and 5 positions respectively.

Looking again at the restricted QK matrices for H1 and H4, I notice both have a small positive self-term in the third dimension (~0.07-0.08), suggesting a term like r² cos(ω(i-j) - θ) plus a small z_i z_j contribution if positions are encoded as a spiral (r cos ωi, r sin ωi, z_i). I'm now checking whether MLP0 or other components might refine this signal further.

Comparing scenarios: if ω≈34° (period ~10.6), H4 would act as a previous-token detector while H1 at 15° offset attends current and previous positions; alternatively, both could be slower local-window attenders. The causal mask doesn't help disambiguate this at small position indices, so I'm considering whether training dynamics from random initialization (std 0.02) might explain why the positional embeddings developed this particular rotational structure.

A char-level LM trained on normal (as-is) text would almost certainly develop a sharp previous-token head in layer 0, since that's such a dominant useful feature—its QK would be positionally dominated and its OV would move full token identity with high rank. H1 fits this profile best: strong positional QK plus the strongest OV (top value 1.94), suggesting it's the previous-token head at a 15° angle, while H4 would then serve as a local head attending 2-3 positions back with weaker OV. For shuffled text, I'd expect H1 to instead function as a recency-weighted local window with strong OV rather than a clean previous-token detector.

I'm also noticing L0H0 has a strong but non-positional OV with positive QK eigenvalues, which looks like a content head attending to similar tokens—possibly spaces or newlines marking word/line starts in as-is text, or tracking repeated-character occurrences in shuffled text.

Thinking through induction structure instead: in a 4-layer model trained on code, I'd expect a classic induction circuit where a later layer's K reads L0H1's previous-token OV output and Q reads the current token embedding, with the QK circuit matching tokens positively. I want to test this by restricting the composed circuit to the top singular vectors of L0H1's OV output space and checking positivity there.

That attempt gave noise too — maybe L0H1 isn't actually a prev-token head, or my read space T is off. Checking L0H4 similarly yields nothing clean either, so there's no clear sign of induction via these heads. I should consider that induction might instead route through L0H0 or via composition with MLP0.

So far I'm finding no detectable induction circuit. If this text were ordinary Python or English, induction should show up clearly, even modestly, in a 4-layer model trained this way — its absence could mean the text is shuffled, where no real induction structure exists. But I should be cautious since my detection power is weak and noisy. To calibrate, I'm looking for any clear "copy-like" QK signature by checking positivity of W_q^T W_k across layers ≥1 — heads that attend to positions with similar residual content, like same-token heads. L1H0 shows strong positivity (+0.50), suggesting that even in shuffled text, heads attending to identical characters could still emerge.

I'm now wondering whether these same-token attention heads are content-based matchers rather than true induction, since they don't rely on positional structure. If a model trained on shuffled text needed to know line boundaries, it would have to infer them purely from content (like detecting newlines), which seems difficult without positional cues — this is getting complicated to pin down. I'm also considering where "local" attention heads in later layers come from for unshuffled text, since character-level models often develop heads that attend to the previous space or boundary token.

That seems plausible. Let me instead check the L0 token-token interaction eigenvalues directly: L0H0, L0H2, and L0H5 show positive values around 0.25-0.33 on non-positional dimensions, suggesting "same token" matching heads, while L0H1's larger value (+0.53) aligns with the positional plane instead.

I should check whether q and k in these layer-0 heads read the same direction (normal, same-token matching) or different directions (non-normal, like any-char attending to space/newline). Looking at the truncation data for H0, H2, H5, most show non-normal behavior with moderate singular value ratios, suggesting these aren't simple same-token matchers but rather something like query-independent attention to specific delimiter tokens.

Now I'm weighing the overall evidence to decide ordering: the as-is condition shows clear relative-position heads at L0 (H1 with small offset, H4 with larger offset) with strong OV weights and heavy downstream composition into L1, resembling a classic previous-token plus two-back induction setup—while the shuffled condition shows no such induction structure and no evidence that later layers read position at all.

I'm digging into whether L1H2's unusually high composition with L0H4 across Q, K, and V simultaneously is really consistent with induction, since induction typically only requires K to read the prior head's output while Q tracks the current token—so this uniform composition across all three is a bit puzzling and worth flagging as unusual.

Actually, I realize L0H4's output might just be a broadly-important residual direction that many heads read generically, since L0H4→H0, H1, H3, H5 all show high composition too—suggesting this isn't a specific circuit signature after all, so composition scores alone aren't discriminating here. I'm shifting instead to examining layer-0 MLP neurons as token detectors, thinking about clustering their input weight rows to see if they cleanly separate by token identity.

Clustering by token wouldn't distinguish corpus type well though—the romaji vs other vocabularies differ only subtly in token count. I wonder if python-style text would show up more clearly through structural cues like newline frequency or indentation patterns, or through heads that count leading whitespace. I'm also considering that the designer likely generated all six corpus/order variants randomly, so a uniform prior across corpora and orders is probably the right assumption, and I'm looking for weight-level signatures a designer might expect to distinguish them, like python's low entropy producing strong induction effects.

Looking at the shuffled model, attention heads seem broader and less sharp, since the loss there is dominated by unigram stats within a line, giving the model less structure to learn and keeping weights closer to initialization. The weight norms I'm comparing (query projections, feedforward layers) don't give a clean answer without a reference baseline, though I'm trying to reason about expected update magnitudes under AdamW over the training run to get a sense of scale.

Factoring in weight decay equilibrium gives an estimate around 0.05, still not quite matching the observed 0.033, and I realize the result is too lr-dependent to pin down confidently. I should try a different angle to identify T — maybe using the LN weights' per-coordinate variation, though that doesn't seem to connect directly either. Let me instead consider whether L0H1 acts as a prev-token head moving token identity, where its OV circuit's top input singular vectors might reveal T.

Looking at the singular value spectrum of L0H1's OV matrix, only about 6-7 dimensions stand out before dropping off, which is far from the full 27-dimensional rank I'd expect from a true previous-token identity head (since bigram statistics would need near-full rank). The fact that it's dominated by a single huge singular value suggests it's tracking something like "was the previous character a space" — a word-boundary feature — rather than full token identity.

Comparing against a noise floor, L0H0 shows about 4 meaningful directions and L0H1 about 6, so layer 0 attention heads are moving roughly 10 distinct token-features total, which seems reasonable. Now I want to think about how MLP0 behaves differently between the as-is and shuffled conditions.

In the as-is case MLP0 is likely computing bigram or trigram features based on the current and previous tokens, but with only a coarse ~6-dimensional prev-token signal, that modeling is limited. I'm not confident I can resolve the angle question purely from these mechanistic details, so I want to try a more concrete test: checking whether the third positional dimension behaves like a monotone "distance from start" signal, and whether head 3's key vector reads strongly from it. I'll compute the 3x3 restricted matrices using a better basis drawn from the top generalized eigenvectors of G0 and G1.



---

I'm noticing some values look suspiciously templated, like repeated magnitudes across different heads, which makes me doubt their reliability. Still, I'm computing H4's rotation angle from its 2D block in dims (0,2), getting roughly 50 degrees, and starting to work out H1's corresponding rotation angle.



---

I'm noticing consistent complex pairs across heads at distinct angles—roughly ±34-37°, ±14-15°, and a larger ±128-144° pair—and wondering if these three angles (~15°, ~35°, ~135°) sit in the same positional plane, possibly corresponding to frequency offsets like 1, 2.3, 9 or scaled versions if ω=35° implies offsets near 0.4, 1, 4.

Trying to pin down ω more rigorously: given |λ|≈0.47 for H4, if positional embeddings dominate roughly half the squared norm in this plane, the implied logit amplitude is quite sharp (~11 logits), giving a very peaked attention distribution depending on ω. But without knowing how much of the embedding norm is positional versus token-based in this specific 2D subspace, I can't nail down ω precisely—just bounding plausible ranges and checking whether H1's implied offset (~0.43 if ω=35°) makes sense as a head that blends current and previous token attention.

Under the slow-ω hypothesis with period ≥128, the angles would translate to offsets of 5, 12, and 48 positions, making all layer-0 heads broad windows with no dedicated previous-token head — implausible for raw text but maybe plausible for shuffled text where heads estimate bag-of-words over varying window sizes. So the ordering question really hinges on resolving ω, which pushes me to think about what the learned positional embedding shape would look like under each scenario.

For prev-token heads to exist in as-is text, I need high-frequency components in the positional embedding, whereas shuffled text wouldn't need sharp relative-position signals and could get by with a smoother, low-frequency curve. If this plane traces a circle at rate ω, causal attention heads rotating by angle θ would attend to offset θ/ω, so I'm checking whether a 135° rotation still makes sense within the context length for plausible ω values.

I can't pin down ω from the attention patterns alone, and looking at MLP0 doesn't help either since its neurons' angular tuning in this plane appears uniform regardless of ω. Position 0 being the window start seems irrelevant here, so I'm trying another angle to constrain the minimum frequency needed.

With only one 2D plane (3 dims), a sharp prev-token head would suffer aliasing at multiples of the period 360/ω unless a third dimension decays sharply -- if the model needed truly sharp relative attention, it should have invested in multiple frequency planes to avoid this. Since we only see one plane in use, that suggests either position is handled coarsely (maybe shuffled) or ω is low enough that the period exceeds the context length, avoiding aliasing within normal windows.

Checking the singular values beyond the top 3 confirms this: H4 drops off after the 3rd dimension (values 1.27, 0.81, 0.62, 0.55, with the 4th barely above noise), and H1 has about 4 meaningful dims. So the positional heads really are only using ~3-4 dims, which is fewer than I'd expect for a sharp previous-token head in a 128-context model relying on a single-frequency helix.

I consider whether the amplitude of token embeddings could be large enough to narrow the effective positional resolution, but that would require an implausibly large logit amplitude, so I rule that out. The positional heads are likely just broad or aliased rather than precisely tuned, which makes me think about what kind of positional attention would actually help if the text were shuffled -- something like "attend within the current line" using recency as a proxy for line boundaries, given typical Python line lengths.

That's a bad prev-token head, so the model would surely learn additional frequency planes to suppress those aliases as training proceeds—the gradient pressure for clean prev-token info is too strong in as-is text, meaning it likely ends up with multiple positional planes rather than one. Alternatively, the positional embedding might be high-dimensional and near-random from init, with the prev-token head implementing the shift structure through a high-rank matrix across the full 128 dimensions, which would show up as elevated singular values spread across many dimensions rather than a clean low-rank signature.

This structure could be hiding inside the noisy bulk of the spectrum, where trained heads already show elevated singular values compared to random baselines—plausible if the learned positional embeddings retain a random-like character from initialization, as tends to happen in small char-level models. Looking at the non-normality ratios, L0H3, L0H0, and L0H2 stand out as candidates for this kind of shift structure, with L0H3 showing the lowest ratio (0.38 vs a random baseline of 0.47) and near-zero eigen positivity, and its key appears to read from the positional plane. I'm now thinking about how to actually test this hypothesis.

If positional codes behave like near-orthogonal random vectors, the resulting matrix would act like a shift operator built from outer products of adjacent position vectors—nilpotent, so eigenvalues near zero but singular values substantial across the ~127 dimensions. Comparing ratios across heads, H3, H4, L1H4, H0, and H2 all sit modestly below the random baseline, but when I restrict to the top-4 singular directions, the gap widens sharply—H3, H2, and H0 drop to 0.23–0.30 while H1 stays at 0.94, versus an expected near-zero ratio for unrelated random subspaces. That suggests the non-normality is concentrated in the top directions for these heads.

I'm realizing this metric alone can't tell apart a genuine shift structure from an unrelated token-to-token or token-to-position mapping, so I need to reconsider the layer-0 input question of how many dimensions are needed when positional and token embeddings have comparable norms. Rather than chasing the shift-matrix idea further, I think the more productive path is revisiting the induction test directly, since induction behavior in intact models is such a strong, distinctive signature—and its disappearance in shuffled models is equally informative. That requires a prev-token head in an early layer writing out previous-token identity, followed by an induction head in a later layer that reads it.

I want to compute the top eigenvalues of the full composition matrix and check if they're large and positive, which would signal a true induction circuit, but the top5 positivity numbers I'm seeing look noisy rather than cleanly diagnostic. I'm reconsidering the approach—maybe I should focus on the pairing where the query reads token identity and the key reads the range of the previous OV output, and compute that directly for the strongest candidates.

Actually, maybe positional info gets picked up by MLP0 instead, which could write positional features into other directions that later layers read. That would explain how "distance from line start" could be computed without direct position access — via attention to newline tokens combined with a slow-decaying recency signal, so a head could attend preferentially to the most recent newline weighted by proximity.

I think the better path is testing induction-head asymmetry directly: checking whether a later query direction strongly reads the output range of an earlier prev-token head's key while the corresponding query projection doesn't, which would be a signature of induction circuitry despite not having direct embedding access.



---

I'm seeing contradictory duplicate entries again (same head pairs with different values), which points to output corruption. I should rerun and print the full 6x6 matrices for fk and fq restricted to L0→L1 only, in a more compact and verifiable format.



---

I want to double-check these numbers in case there was corruption, but they seem plausible. I'm tracing how L0H4's output feeds into several L1 heads' keys and queries with varying weights, and similarly mapping out where L0H1's output gets read downstream.

Looking at L1H4, I notice it reads L0H0, L0H2, and L0H5 outputs with similar weights around 0.13-0.14, suggesting those three heads might write to a shared direction in the residual stream.

This pattern — where K-side composition exists but Q-side doesn't involve the same L0 head — resembles classic induction circuitry: L0H4 as a previous-token head feeding keys, while queries come from the current token embedding directly. I'm wondering whether in shuffled text this same K-composition with a local-window head carries the same meaning, since the usual "previous token" logic doesn't map as cleanly onto scrambled sequences.

So I want to check what L1H1, H2, and H5's queries actually attend to — likely the embedding or MLP0 output rather than another L0 head, consistent with the induction-head story where the query reads the current token itself. I'm going to run a bilinear form test between the Q-input and K-input for L1H2, the strongest candidate, to confirm this matching behavior directly.

To test whether this is truly induction-like, I want to examine the singular vectors of the composition matrix C — checking if the top left singular vectors (representing query-side directions) align closely with the right singular vectors (representing the token features that L0H4 reads at the attended position). If the cosine similarity between corresponding pairs is high, that's strong evidence for the induction mechanism, so I'll compute those cosines along with the eigenvalue positivity restricted to that subspace.



---

Looking at the top singular component for L0H4→L1H2, the cosines across heads (-0.25, -0.31, +0.38) suggest query and key directions that are only partially correlated, rather than the near-perfect alignment (cos≈+1) expected from classic identity-like induction copying. This points toward a single feature, maybe something like "space," driving the interaction rather than full-token copying.

The negative signs for H2 and H1 suggest an anti-matching pattern, like attending away from positions whose previous character shares a feature. Given a 4-layer model, I'd expect induction circuits around layer 1 or 2, and the cosine-of-singular-vectors method should reveal identity-like structure if it existed—since token embeddings forming a symmetric PSD matrix would naturally produce query/key singular vectors with cos≈+1. Since that's not what I'm seeing, L0H4 doesn't appear to feed an induction mechanism in L1 this way, so I should check whether the induction head instead relies on MLP0-processed token information.

I'm now testing all combinations of L0 attention heads feeding into L1 and L2 heads, checking the cosine similarity of the top singular triplet and positivity of the restricted top-6 subspace, then printing a compact matrix comparing L0→L1, L0→L2, L1→L2, and L1→L3.



---

I'm not finding clear identity-like induction QK circuits, which surprises me given that a 4-layer model trained on 1B characters should develop induction heads, especially for highly repetitive text like Python code, and even char-level English/romaji models typically show this after sufficient training.

Checking the singular value spectrum of the composed circuit confirms this — the top values (0.041, 0.027, 0.024, 0.021...) look noise-level rather than showing the ~27 elevated singular values with matching singular vectors that a clean induction circuit would produce. I also check whether previous-token information routes through L0H1 instead, but its output row shows no clean directional alignment either, so prev-token heads seem absent too — all of this is pointing toward the shuffled hypothesis.

A shuffled model would plausibly show layer-0 positional/recency windows plus content-matching heads, with later layers operating purely on content and ignoring position — which matches what I'm seeing: L0 positional code is smooth and only ~3-dimensional, while L0H0/H2/H5 behave like same-token content heads. An as-is character model would typically carry a richer, higher-dimensional positional code into later layers for things like prev-token or 2-back attention, which I'm not finding. This pushes my estimate toward shuffled, around 60-65%, though I want to double-check the low-dimensionality claim about the positional code before settling on that.

The generalized eigenvalue spectrum comparing L0 QK against later layers shows just 3 standout values (1.22, 0.85, 0.81), then a smooth gradual decay (0.67, 0.58, 0.55, 0.52...) against a median around 0.20 — so there's no clean cutoff suggesting a rich ~30-dimensional positional code exclusive to L0, but the tail is elevated enough that maybe 10-20 dims carry weak positional signal, which complicates a clean read. I want to look more closely at L0 QK for structure beyond the 3-dim plane, particularly whether there's non-normal, shift-like behavior in weaker heads like H3.

Checking the OV side instead: a true "previous-token" head should move most of the token identity through a high-rank OV, but L0H1 only has ~6 elevated dims and L0H4's OV is nearly at noise level (0.75 top singular value). That's odd — if L0H4 were the primary previous-token head, I'd expect it to have a strong OV, so its weak output suggests it's playing a more purely positional role rather than actually carrying content forward.

But for shuffled text, I'd expect the model to care about the bag of chars in the line rather than order, which would mean moving full token identity broadly — so a high-rank OV head counting chars. Yet the elevated-but-not-huge singular values I'm seeing (like 0.48, 0.42, 0.36 vs random baseline 0.33, 0.29, 0.27) suggest broad, mildly-elevated structure across many dimensions rather than a few dominant ones, which is actually consistent with moving many features at once.

I can't firmly conclude this yet, so I'm trying another angle: comparing the high-rank learned component of OV across L0 heads versus later heads to see if there's a meaningful difference, and considering whether a dominant input direction in one head's OV (like L0H1's v1) also shows up as the top singular direction elsewhere — in W_fc.0, or read by Wq.0/Wk.0 — which would suggest it's a shared, broadly-read feature.

Now I'm checking whether v1 corresponds to something like a "space vs letter" or "newline" distinction by computing overlaps between the top input direction of W_fc.0, the top input directions of OV_{L0H1} and OV_{L0H0}, and the positional plane.



---

The top input directions of L0H1's and L0H0's OV align closely with fc0's dominant directions, suggesting a shared token-level feature like space-vs-letter or vowel-vs-consonant that MLP0 reads and both heads move. I want to find a common basis spanning these dominant directions and figure out what specific feature it represents.

I'm also reconsidering the previous-token hypothesis for L0H1: a small positional rotation plus a strong OV projection onto this dominant feature would mean it's mainly shuttling the previous token's class rather than doing genuine positional work. Separately, I'm looking at the third dimension of the positional code — if the embedding is a helix combining a rotating circle with a linear drift, the slight positive self-terms in QK along that drift axis would bias attention toward nearby positions sharing similar drift coordinates, which matters for pinning down the rotation frequency ω.

If ω is small, the whole circle behaves like a slow curve resembling low-frequency Fourier components typical of smooth learned embeddings, giving broad attention windows; if ω is large, the circle cycles many times over the sequence, which would be an unusual structure since smooth models tend to favor low frequencies dominating the spectrum rather than one isolated high-frequency plane — though I'm wondering whether a model that specifically needs a previous-token head might instead learn to carve out just such a high-frequency circle from its initially random embedding.

Looking at known cases like GPT-2's positional helix, the structure supporting prev-token attention tends to stay low-frequency with sharpness coming from large norms rather than high ω. I'm trying to figure out whether the two heads here with clean positional structure, H1 and H4, each rely on a single dominant plane, and whether their required sharpness pushes toward larger ω and amplitude despite the aliasing risk at certain offsets, something that might be softened by a secondary drifting dimension and by softmax competition between heads.

Trying to pin down the actual amplitude contributed by the third dimension, which drifts roughly linearly, I'm checking whether its score contribution depends only on each token's own position rather than the relative offset, making it less useful for pure relative attention. I'm also comparing the eigenvalue magnitudes for H4 and H1 to get a rough sense of scale, though that alone doesn't settle it, so I'm considering whether layer 0's MLP might be handling absolute position instead.

Shifting focus, I'm thinking about shuffled-text-specific mechanisms the model might rely on, like detecting line boundaries via newline tokens, since position-since-newline matters for predicting line length and spacing. This points toward a possible "current line" attention head that restricts itself using line identity encoded by an earlier layer that tracks the most recent newline.

Specifically, L1H0 stands out with a strongly normal positive QK pattern, and its query draws from L0H1's output, which itself is a positional local head. I'm now examining whether this reflects a same-line attention mechanism, where line-start codes computed in layer 0 let layer 1 heads compare positions within the same line, while also noting that similar normal-positive QK behavior shows up in as-is text for same-token attention. I want to dig deeper into what exactly L1H0 is doing.

I'm stepping back to weigh my heuristics here: I haven't found an identity-like induction circuit across the layer pairs I tested, which is notable since induction circuits are typically expected in autoregressive models, though I'm unsure how sensitive my test actually is. I'm thinking about ways to validate the test's sensitivity, perhaps by checking for known structures like same-token heads, to make sure I'm not missing something real.

Given that previous-token information might be distributed across multiple heads or routed through the MLP, a single-head test could be too weak to detect induction. I'm planning to instead test using the combined OV output of all layer-0 heads summed together, checking for restricted positivity and alignment of the top singular vectors against layer-1 and layer-2 query-key products.



---

L1H0 stands out with a strong self-matching score in the QK analysis, while the rest look fairly weak, and the OV induction test comes up empty. I'm now questioning whether it even makes sense for induction QK to behave like an identity mapping in the residual basis, so I'm working through the algebra of how the query reads the current token embedding and the key reads the OV output applied to an earlier token, trying to pin down what the combined circuit matrix looks like.

If the previous-token signal comes directly from an L0 head and the query relies on the raw embedding, my test logic holds, meaning none of the L0-sourced prev-token heads form induction circuits here. That makes me wonder if the prev-token head actually sits in L1 with induction happening in L2 or L3 instead—L1→L2 showed nothing, so I should now check L1→L3, L2→L3, and also try Q-composition variants where the query itself is built through a prev-token head rather than the key.



---

The uniform +0.03 values look suspiciously templated, like the channel might be fabricating output rather than reflecting real signal, so I can't trust it even though nothing else stands out. I'm wondering if the lack of evidence stems from my method itself, so I want to sanity check it against a synthetic induction circuit with comparable noise levels — building something like random embedding vectors and a prev-token OV circuit to test the approach.

But I realize a bigger issue: the induction head's query might actually be reading through MLP0's output rather than the raw embedding, since layer-0 MLPs often act as an extended embedding in small models — this would make my linear/identity-based test blind to the real structure, since the mapping becomes nonlinear. That weakens any conclusion I'd draw from the absence of identity-like patterns, so I should instead think about this from the angle of the loss surface under shuffled training data, where any order-sensitive function would behave differently.

This leads me to consider permutation invariance: if order truly doesn't matter in the shuffled setting, attention shouldn't depend on fine relative position, yet H1's 15° and H4's 35° rotations in L0 look like purposeful offset-specific attention. Under a shuffled model I'd expect only generic recency or no positional structure, not a window centered several positions back excluding the nearest token — that's hard to justify as useful under shuffling. Whereas under the natural, unshuffled interpretation, H1 attending to the previous token and H4 attending 2-3 back is exactly the kind of n-gram context mechanism I'd expect, with their OV circuits moving character-class features like spaces.

I'm now trying to pin down ω by reasoning about L0H3's rotation angle and what offset distances different ω values would imply, comparing plausible offset sets (small ω giving broad distant windows vs larger ω giving tight nearby windows) to judge which is more consistent with a coherent set of positional heads, and considering whether a second harmonic plane at 2ω might help sharpen the estimate.

The doubling hypothesis doesn't quite hold up cleanly across the two heads. Let me instead directly compute the shared planes between H1 and H4 by taking the joint top-8 subspace of their q/k vectors and doing a restricted eigen-decomposition for each head there.



---

I'm noticing H1's peaks at 14° and 41° look suspiciously close to a 3x harmonic relationship, while H4 shows peaks at 37° and 62° instead—trying to work out if these offsets map to different frequencies between the two planes.

The ratios don't match cleanly (2.9 vs 1.7), suggesting these two heads aren't using the same positional mapping, which could just be noise rather than a real signal. I'm wondering if I need a more global test instead—maybe looking at how MLP0 engages with the positional plane differently between as-is and shuffled models, since newline prediction in shuffled text seems to require tracking absolute distance since the last newline in a way the as-is model wouldn't. Without clearer evidence though, I can't fully resolve this from the data I have.

I'm wondering whether high-rank QK could reflect bigram-like matching rather than identity matching—testing that found only L1H0 showed strong same-feature correlation, with others weak. I'm also considering whether positional heads tracking recency or newline structure play a bigger role in shuffled text, since spacing becomes essentially random there.

I want to check whether MLP0 reads prev-token features from L0H1's output to encode bigram statistics, and whether that pattern differs between as-is and shuffled text—though it's unclear it would differ much. I'm also starting to think about copy heads in later layers, where OV circuits might map token identity directly to output predictions.

Turning to the corpus question directly, Python's distinctive feature is indentation—runs of spaces at line starts in multiples of four, plus short lines—suggesting positional heads tied to newlines, whereas English has long lines where newlines are rare, and romaji falls somewhere in between. But without embeddings, I can't actually identify the newline token, and the dominant feature driven by L0H0/L0H1 is likely just "space," the most frequent character across all three corpora at varying rates. I'm not finding a clear path to distinguishing the corpus from this angle.

Shifting to romaji specifically, it has strong consonant-vowel alternation with vowel identity as the dominant feature, more pronounced than in English. For Python there are identifiers, keywords, heavy use of spaces and newlines. I'm also wondering whether certain Latin letters like l, q, v, x are entirely absent from the romaji corpus, since it's likely generated by romanizing Japanese text via a tool like pykakasi or cutlet—meaning foreign words in katakana get converted to Japanese syllables, while any stray Latin letters from the original text would remain untouched.

Without embedding access I can't directly probe how many directions the embedding space spans, so I'm thinking instead about average line length: Python lines run short (~30 chars), Aozora romaji lines are much longer due to paragraph-per-line formatting (150-250 chars), and English Wikipedia paragraphs are longer still (~600 chars). This matters because in shuffled text, the model's main signal is the content within the current 128-token window—for English, the whole window is essentially one long line requiring broad attention, whereas Python's short lines mean several lines fit per window, forcing the model to localize attention to the current line.

So I'm reasoning about which corpus would produce a model where the low-dimensional positional code in layer 0 stops mattering in later layers. If English shuffled lines are long enough that one line fills the whole window, there's no real need for positional attention at all—just uniform unigram-style attention. But we actually observe substantial positional QK structure in heads like H1/H4, which argues against English shuffled, since newlines there are rare (roughly one per 600 chars) and wouldn't demand this kind of newline-relative position tracking that we're seeing.

In contrast, with python shuffled where lines are short, positional and newline structure becomes essential—attention to the current line, recency-based heads finding the last newline, broad local windows matching line length. That fits the observed pattern much better, so I'm narrowing the candidates down to python-shuffled or any as-is corpus, and now I want to check whether later layers show corroborating evidence.

Checking whether L1H0's K also reads L0H1's output the way Q does—if a same-line matcher should read the same line-code symmetrically on both sides, but here Q reads it strongly while K barely does, that's inconsistent with the hypothesis. I'm also mindful I've burned through roughly 40 of my 100 runs, so I should be more strategic—maybe trying to isolate ω through the L0 MLP or exploiting how LayerNorm mixes token and position information to see if I can decouple them analytically instead of empirically.

Trying to bound the attention logit amplitude using the LayerNorm output norm and the fraction of residual variance sitting in the positional subspace, I compute rough max amplitudes for different variance fractions and eigenvalues, then check whether a previous-token head's score at distance 1 could plausibly exceed the scores at distance 0 and 2 by enough logits to work.

Given H4's angle of roughly 35°, the amplitude needed to clear the previous-token threshold would require an unrealistically high fraction of positional variance, so H4 likely isn't a sharp previous-token head—its attention would instead spread out softly across nearby positions rather than peaking sharply at distance 1.

Maybe it's implemented via high-dimensional near-orthogonal position codes with a shift operator hidden in the bulk, which would make the QK eigenvalues nearly nilpotent with broadly elevated singular values — consistent with L0H3's low |λ|/σ ratio. I'm testing whether a nilpotent shift structure (where traces of powers vanish) could explain this, but it's not clearly distinguishable from a random matrix's trace behavior either.

If instead the positional codes are random-like and used for sharp previous-token attention, the relevant subspace would be high-dimensional, meaning many generalized eigenvalues should be elevated. The distribution I found shows three clear peaks followed by a gradual tail, with a median around 0.20 — so I want to check how many values sit above roughly twice the median to estimate the true dimensionality of the positional code, and I'm pulling quantiles to see this more precisely.



---

The repeated line looks like channel corruption, but the first shows a smooth decline rather than a plateau of elevated dimensions, suggesting the positional code is low-dimensional with only about 3 clear dims, while a richer random-like position code isn't obviously evident—though it could be masked by noise mixing at L0 QK. I want to compute a baseline for what this generalized eigenvalue distribution would look like under a random null to compare against.

Working through the Wishart ratio math, I estimate that the observed spectrum is mostly random-like except for the top 3 eigenvalues, reinforcing that L0's positional machinery is essentially a 2D rotation plane plus one extra dimension. That's striking because such a low-dimensional code should make sharp previous-token attention geometrically difficult, yet a char-level model trained on a billion characters should surely need sharp previous-token information somehow—so either I'm missing a mechanism, or the low-dim circle is compensated by large amplitude and a third drift dimension that resolves the aliasing.

Testing this numerically, with a reasonable fraction of the LN-norm concentrated in the plane, the resulting amplitude does give sharp previous-token attention at the primary offset but also creates strong aliasing near a multiple of that offset where the angle wraps back around. For the third dimension to fix this, it would need to contribute a penalty of several logits at that aliased distance, but if the third dimension is just a monotone drift term, its contribution doesn't naturally decrease with distance the way it needs to—so I'm not yet convinced the aliasing problem is actually resolved by this mechanism.

Working through the rotation offsets, slow ω would produce windows centered around 5 and 12 positions back, which could correspond to sampling recent characters like the current line in shuffled code with typical line lengths of 30-40 characters. That's plausible for both heads being local/broad context samplers, though I'm still trying to figure out where the prev-token head would be in the as-is text case.

I'm leaning toward the shuffled interpretation at maybe 60% confidence now, but want to double-check the eigenvalue magnitude reasoning — specifically whether the positional embedding norm dominates over the token embedding norm after LayerNorm, since that would determine how much of the normalized vector lives in the positional plane. The MLP0 readout values (2.70 vs 1.43 baseline) only show a modest signal, and I'm also reconsidering the 135-144° angle range for the third head.

That head seems to query on content while its keys read position — maybe implementing something like "attend to early positions." This makes me wonder whether later layers could build a "line identity" mechanism for shuffled text, allowing broad attention within a line via bag-of-features. But since later layers don't appear to read the positional subspace at all, they can't do recency-based restriction, so their attention must instead be purely content-driven across the whole context window rather than confined to the current line.

So maybe line restriction happens entirely in layer 0 through the recency-window heads, with later layers operating on content alone regardless of order. For normal, unshuffled text, induction heads wouldn't need positional info, but other common heads like "attend to the previous word" or "most recent delimiter" would need recency signals — yet this doesn't give a decisive answer either way. I'm trying to think of a cleaner test: maybe check whether the model's output is invariant to reordering characters within a line for shuffled text, versus being highly sensitive to the last token's identity for normal text, to see if the weights reveal this asymmetry directly.

In as-is text, I'd expect MLP0 to function like a bigram table that reads the current token heavily, with large singular values in its weight input direction; in shuffled text, MLP0 would instead need to read the aggregated attention output representing the "bag" of line tokens rather than the current token's identity alone. I could measure this by comparing how much W_fc.0 reads from the attention-0 output subspace versus the embedding subspace — and it looks like W_fc.0's top input directions actually align closely with the OV input directions of L0H1 and L0H0, token-level features, with a cosine similarity around 0.6.



---

MLP0's top neurons draw roughly equally from attention-0 outputs (29%) and token inputs (31%), with little overlap between these subspaces, suggesting it's merging current-token info with attention-moved features rather than making a decision itself. Now I'm reasoning about the "without replacement" constraint in shuffled text — trying to work out whether the bag of remaining characters in the line is really the dominant signal for predicting the next character once some characters have already been seen.

Actually, since the seen characters are a random sample from the line's multiset, their empirical distribution estimates the remaining distribution, meaning prediction roughly reduces to copying from characters that already appeared in the current line — a bag-copy mechanism via broad attention heads with OV copying token identity into output logits. In the as-is (unshuffled) case this copy role is played by induction heads instead. So both settings rely on copy-OV circuits, but differ in their QK behavior: broad/query-independent attention restricted to the line in the shuffled case versus sharp previous-token matching in the as-is case — which gives me a concrete thing to test, whether QK patterns are actually query-independent in the shuffled condition.

I'm checking whether this query-independence holds in later layers — in the shuffled setting, copy heads should attend roughly uniformly across the line regardless of the query token, implying near-zero QK response to content (aside from position), whereas layers 1-3 show measurable structure (participation ratio ~31 versus ~48 for random), and L1H0 shows strong same-character self-matching. I'm puzzled by what self-matching would accomplish in shuffled text, since attending to prior occurrences of the current character doesn't obviously help estimate counts for all other characters; this behavior makes more sense in as-is text, where matching the current character sets up the first step of induction — finding earlier occurrences of the current token so a later step can read off what followed it, which typically needs K-composition with a previous-token head.

Since causality prevents a key at position j from directly surfacing information from position j+1, I suspect L1H0's same-character attention retrieves what preceded earlier occurrences of the current character via the residual stream, by reading what L0's previous-token head already moved there — though I'm unsure how useful that really is. I also consider whether L1H0 functions more like a "same line" head that attends to similar contextual features rather than literally the same character, and I want to examine how L1H0's query interacts with L0H1's output, since L0H1 seems to move primarily one dominant feature through a local recency window — possibly something like "is there a newline nearby." I plan to compute L1H0's QK matrix restricted to L0H1's top singular directions to test this directly.



---

I notice L1H0's dominant QK component is essentially a self-matching pattern on L0H1's output direction, creating a rank-1 score where positions with similar values of that feature attend to each other—suggesting it's picking up on some shared local property across positions.

Since L0H1 looks like a positional head reading a local window to move a single dominant feature (possibly "space" or "newline"), I'm wondering if L1H0 then matches on this to create an "attend to word starts" type behavior, similar to patterns seen in character-level models, but I'm not fully certain yet—it could also just be reading local space/newline density rather than something more semantic. I want to check whether L0H1's output direction aligns with the top direction of fc0, since both L0H0 and L0H1 show similar cosine similarities there, hinting they might be moving the same underlying "space" feature.

I'm trying to pin down whether the positional circle is slow or high-frequency, since that determines whether these heads function as broad local windows or something closer to periodic attention—if it's slow, these look like true local heads spanning several characters back. The real question is whether there's a hidden previous-token mechanism buried in the higher-dimensional positional code outside the main 3D plane, so I want to test whether that orthogonal subspace has the structural signatures of a shift operation.

My reasoning is that if q reads position i and k reads position i-1 within a shared random-like subspace P, then the q-read and k-read subspaces should be nearly identical even though the mapping between them is a non-normal shift with small eigenvalues. I'm now comparing the singular subspace overlap between Wq and Wk against what random chance would predict, and looking at how it relates to eigenvalue-to-singular-value ratios across the heads—seeing q1-k1 showing a notably higher overlap (0.62) than most others, with q3-k3 lowest at 0.11.



---

The q-k overlap seems unremarkable across heads, just a natural consequence of shared residual stream structure, with H3 being the lowest. I'm now weighing the evidence around positional structure in layer 0, which appears low-dimensional—essentially a rotation plane plus one dimension—and gets used by H1 at a small angle, H4 at a larger one, and H3's key.

I'm not seeing this positional code reappear in later layers, nor any high-dimensional positional encoding, nor induction-like identity circuits; instead L1H0 shows rank-1 self-matching onto L0H1's output feature. This fits a training regime where only coarse recency matters rather than sharp previous-token tracking—since the text is shuffled, a single slow rotation plane is enough to create attention windows spanning several tokens back, and a nonzero rotation angle produces a fuller centered window compared to the half-window you'd get with zero rotation.

Checking this against the actual eigenvalue spectra for H1 and H4 confirms a single dominant positional plane in each, with the remaining eigenvalues near noise level—so I'm fairly confident this is the shuffled-text scenario, maybe 65% likely, though I want to keep examining the drift dimension in the joint subspace since H1 shows a notable real eigenvalue there too.

Thinking about which corpus is involved, I'm now weighing line-structure differences: Wikipedia paragraphs versus short section headers, against Aozora romaji texts with lots of short dialogue lines interspersed with longer prose.

If the lines are long and shuffled, the window is basically one bag of characters, so the optimal attention pattern would be uniform rather than recency-biased — meaning narrow positional windows of 5-12 tokens back would actually be suboptimal compared to attending uniformly across all 128 positions. That the model instead shows localized recency windows suggests line boundaries are frequent, pointing toward short-line corpora like Python code (where indentation and line shuffling create short local structure) rather than long-paragraph English text.

I'm checking whether OV spectra of later heads show a plateau corresponding to the number of distinct characters used, but the singular values decay smoothly instead of plateauing, so that signal isn't there. I'm shifting to think about structural differences in MLP activity instead — Python's repeated tokens and indentation versus English's varied vocabulary.

Romaji's CV syllable structure matters in the as-is ordering but collapses to just unigram character frequencies once lines are shuffled, so that distinction feels weak as explanatory evidence. I'm now considering the dominant layer-0 token feature read by MLP0 — likely space/newline tokens in normalized Python versus vowel/consonant balance in romaji — as a possible driver of the effect.

Romaji's strong CV alternation would give as-is text sharp prev-token heads in layer 0, while shuffling destroys that local structure entirely. I'm leaning toward shuffled order with probability around 0.6, though I still need to pin down the corpus — trying to bound how large an attention logit gap between adjacent positions a layer-0 head could realistically produce given the unknown positional encoding beyond the identified plane.

With near-random positional embeddings, building a genuine prev-token head would require learning a complex rank-128 mapping across them, which seems slow to emerge — models likely settle into smoother positional structure well before 30k steps, which shapes my expectation for what's happening in this nanoGPT character-level Shakespeare setup.

Checking H3 more closely: its key vector overlaps moderately with other positional-adjacent heads (0.27-0.40) while its query doesn't show that pattern, which is a weak signal of an asymmetric content-vs-position read but not conclusive on its own.

I'm also considering L0 heads H0/H2/H5, which seem to have mild positive QK alignment suggesting they're matching repeated characters — relevant for the shuffled-text case since repeated letters in a line (especially without replacement) could bias predictions toward or away from characters already seen.

As-is layer-0 heads more likely follow local positional patterns (previous token, 2-back, space/newline tracking), which is less typical in shuffled text. Weighing the corpora, I'd lean toward shuffled being around 0.62 overall, with python's short lines favoring strong positional structure, English's long lines reducing the need for such heads, and romaji falling somewhere in between due to its mix of short dialogue lines — though I'm not fully settled between python, romaji, and english here.

I'm also second-guessing myself on whether later layers actually ignore the positional plane, since as-is python's indentation requires counting spaces from the line start, which relative-position heads often handle, while shuffled-python's newline prediction could instead rely on a head tracking distance since the last newline. I want to pin down what L0H1's top feature actually represents — possibly a newline indicator — since that would explain why L1H0 matches so strongly onto it (0.141 vs. 0.075 for the next closest).

I'm checking whether L0H1's top output direction gets read by other L0 heads' keys or queries, since if it encodes "newline," a content head attending back to newlines with a position-independent query would confirm that interpretation.



---

I'm noticing that v1, the dominant feature moved by L0H1 and aligned with fc0's main input direction, is read strongly by keys across several heads (H0, H2, H3, H4) at roughly 5x typical magnitude, suggesting it's a high-variance token feature like "is space" or "is newline" that many heads key on.

Trying to reason through whether this corresponds to word boundaries (as-is text) or line boundaries (shuffled text), I suspect space is the likely candidate since it's the most frequent character, but I'm not confident without more digging. I decide to try one more concrete check: working through the layer-0 layernorm math to see if the positional component's top eigenvalue can explain a sharp previous-token attention pattern, tracking through the norm of the pre-gain LN output and the rotation matrix's leading eigenvalue in the positional subspace.

I'm estimating the fraction of energy the positional signal occupies within that 2D subspace relative to the full embedding, and testing what logit amplitude that implies. Given plausible fractions (roughly 0.1-0.3), the resulting amplitude of 2-7 logits is too weak to produce genuinely sharp previous-token attention, suggesting H1/H4 are soft windows rather than true previous-token heads — which leans toward favoring the shuffled-text hypothesis, though I remain unsure since a higher-dimensional, less visible structure could still be at play.

I check this by comparing eigenvalue-to-singular-value ratios across heads, since a true shift-based head should show ratios near zero. The values I'm seeing across layers 0-3 range narrowly between 0.38 and 0.47, with L0H3 only marginally lower than the rest — not the sharp drop I'd expect if shift dominance were really present.

I also note the trace positivity for L0H3 is barely above zero, suggesting it's at most a weak positional head rather than a clean prev-token mechanism. This makes me reconsider whether rotation structure in a positional plane would even emerge from shuffled text, versus whether it's really a signature unique to as-is models using soft local windows over recent characters.

I'm weighing whether H4's partial eigenvalue drift reflects an aliased but still functional prev-token head tolerated in a well-trained as-is model that's had 30k steps to develop richer frequency content, versus a shuffled model lacking any reason to encode a slow-frequency rotation rather than a simple monotone position coordinate. I'm leaning toward the shuffled hypothesis at around 0.62 confidence, but I'm still unsure if the positional plane's rotation really distinguishes the two cases cleanly.

Considering how MLP0 might read the position plane in a shuffled-data newline-prediction task, I think the model likely needs something like "distance from the last newline," which could be computed via attention to the newline token or an averaged positional signal, though the value-vector evidence for this is only mildly elevated and not conclusive. I'm now trying to figure out which corpus — likely Python given its short lines — would most favor these local-window and newline-tracking features.

For shuffled English text, lines run long, so local recency windows seem less necessary since broader bag-of-words heads already capture similar information redundantly; the presence of recency-window heads among the most structured L0 heads instead points toward a corpus with frequent, short line breaks, like Python or Japanese romaji. Thinking about Aozora-style Japanese novels, paragraphs and dialogue lines tend to be short, often just one to three sentences, which could also explain why such windowed features would be useful there.

Given typical character counts—Japanese sentences around 40 characters (roughly 100 in romaji), English Wikipedia paragraphs spanning 500-1000 characters, and Python lines around 30-40 characters—line structure seems to matter most for Python, somewhat for romaji, and least for English. The recency window offsets and widths observed (roughly 5-25 characters back) line up well with Python's short line lengths rather than English's longer paragraph structure, pushing my probability estimate toward Python at around 0.45, though the un-shuffled corpus case remains more ambiguous.

I'm reconsidering whether to weight the as-is condition higher, thinking through how keys in later layer heads might compose with the positional window feature from an earlier layer head—specifically whether this could serve to locate line-start positions by detecting a newline-proximity signal, which would still make sense even in shuffled text if the mechanism is about finding recent line boundaries rather than relying on genuine long-range structure.

I'm also weighing whether this top singular component reflects a single interpretable feature direction (like a query-key alignment tied to word-start detection) versus something more diffuse, and whether later layers truly avoid reading absolute position or instead inherit positional signal indirectly through earlier-layer outputs—which matters for judging if shuffled-induction behavior could still masquerade as coherent processing.

Given that induction heads reliably emerge in small char-level models trained on English text, I'm suspicious my earlier identity-test found none, since the test might fail if the query depends on MLP0's output rather than raw embeddings. I'm considering running a quick synthetic sensitivity check—constructing normalized random embeddings and a synthetic induction head using a real previous-token OV circuit—to see if the identity test would actually detect such a head if one existed.



---

The synthetic induction circuit yields mean cosine similarities around 0.4-0.55 for the top-5 singular pairs, whereas real weight pairs cluster near zero, with just a few outliers near ±0.3 on the top component. This confirms the test picks up identity-like induction circuits with raw-embedding queries and direct OV keys, but real weights don't exhibit this pattern.

My synthetic setup is idealized though — random embeddings with a clean learned component — so real models could show weaker or messier versions of the structure, yet the near-zero averages suggest no such direct circuit exists between L0 and L1 or L0 and L2. I want to double check L1→L2, L1→L3, and L2→L3 as well since that earlier output looked suspiciously templated, so I'll rerun with a cleaner summary showing, for each layer pair, the max mean cosine and which singular vector pair achieves it.



---

The max mean-cosine similarity I'm finding is around 0.11, far below the 0.4-0.55 range seen in synthetic cases, so I conclude there's no identity-like K-composition induction circuit present anywhere. I should also check whether a Q-composition variant exists instead, where the query looks for positions whose token matches the previous token rather than the standard mechanism.

Actually that alternative doesn't work mechanistically since the OV output would need to come from a position that isn't accessible, so only the standard K-composition pathway makes sense. I check self-similarity on the raw residual for an alternative "same-token" signal and find L1H0 at 0.48 while others sit around 0.08-0.2, suggesting a narrow rank-1 feature rather than a broad content-matching mechanism. Given the lack of induction circuitry, I'm now weighing how this affects predictions across domains—Python code, English Wikipedia, and Romaji—since all of these naturally have repeated identifiers or words that induction would normally exploit, and a 4-layer model trained on a billion characters would almost certainly...

...develop induction heads under normal circumstances, yet my sensitive test turns up none, and combined with the absence of sharp previous-token heads, I'm leaning toward the shuffled hypothesis at around 0.75 confidence. I caveat this with the possibility that queries use MLP0-processed representations which would defeat my test, and I've also checked the summed OV across L0 heads without finding anything. Now I'm going through head-by-head classification for the shuffled hypothesis, starting with L0H0 as possibly a "broad" content head attending to features like spaces or newlines, and L0H1 as a "local" positional head with small rotations attending to nearby positions.

I continue classifying L0H2 through L0H5 and the layer-1 heads as either "broad" or "local" based on their angle and content sensitivity, with L0H4 showing a moderate rotation suggesting it reaches a few characters further back than H1, and the layer-1 heads mostly lacking positional reading or induction behavior, making them "broad." I pause to reconsider whether L0H1 should instead be called "previous" given its small angle, but under the shuffled hypothesis I settle on calling it a recency window, so "local" fits better.

I shift to thinking about distinguishing corpora by structural features rather than head labels — comparing how shuffled Python, English, and romaji text differ in newline frequency, line length, indentation patterns, and space/vowel fraction, since these statistics could reveal which corpus a window belongs to even after shuffling.

Also checking the frequency of such boundary cues further supports that line breaks occur often in this text.



---

I've run enough checks, so I'll finalize my estimate: roughly 45% python, 30% romaji, 25% english for the language guess, and about 30% as-is versus 70% shuffled for the ordering.

