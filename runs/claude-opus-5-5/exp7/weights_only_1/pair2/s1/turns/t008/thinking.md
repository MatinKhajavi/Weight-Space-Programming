The eigenvalues come out roughly symmetric across heads, so there's no obvious copy asymmetry. The large ratio to the bulk looks like it comes from the massive dimensions, probably dims 163 and 400, which give a ±eigenvalue pair. I need to exclude those massive dimensions before reading the spectrum.

I'm moving away from the massive-dimension analysis and looking at the update itself. Both models presumably moved from the same base with Adam steps, so I'm checking the per-element differences in bf16 units. The weights are around 0.04 in magnitude, which puts one ulp near 2.4e-4, and that means the typical difference is about one ulp per element.

Since the base sits exactly on the bf16 grid, each element only moves when its update exceeds about half an ulp. Smaller-magnitude weights have finer ulps, so they flip more readily, which would explain why down_proj shows fewer equal elements.

Comparing the two models' weights element by element, I'm working out the bf16 grid: if their difference is two ulps, the base has to sit at the midpoint, but a one-ulp difference doesn't say which model moved. I'm now wondering whether one fine-tune simply moved more elements than the other.

I'm now considering whether estimating the base from pairs that differ by two ulps would help. That only gives the sign of each model's move at those elements, which is just half the difference I already have. To see each model's update separately I'd need the shared base, and the difference alone cancels that out.

I'm checking whether the per-head difference lines up with each head's existing weights, comparing the cosine against the random-noise floor of about 0.0025. The observed values sit a few sigma from that floor, so I want the symmetric version, cos(A-B, A+B), per head and per MLP unit. I'm also thinking about how the task was likely set up: a Gemma 3 270M fine-tune on sycophancy versus correctness.

I'm weighing whether the intended signal lives in specific heads' QK/OV circuits, since a full-matrix spectral view is swamped by the massive dimensions. So I'm planning to exclude dims 163 and 400, and any others with large post-norm gains, before reading anything off the spectrum. I also want to whiten the residual stream properly, since it's built from scaled, RMS-normalized sublayer outputs.

I'm estimating each residual dimension's scale from the post-norm gains, so that s_i² grows with the embedding and per-layer contributions, and the massive dims stand out only modestly. Then I'll whiten the residual stream with those scales so the weight differences can be compared in those coordinates.

I'm checking whether a copy test in whitened coordinates would even work, since the answer-token subspace isn't known yet. The next idea is to pull that subspace out of the data by looking at shared reader directions in late layers, with dims 239 and 54 as candidates.

I'm thinking about whether the answer-direction signal lives in the late writers, since at the final layers the gradient is proportional to the unembedding applied to the target minus the predicted distribution. For the sycophantic model that target is the user's guess, for the correct model it's the true answer, so the difference should show up as different token directions in those writers.

I'm working out where the asymmetry actually sits: the syc model copies the guess X from context through attention and OV, while the correct model has to recall Y from MLP knowledge and question content. The output directions of the late writers should span answer-token directions for both X and Y, so the copying test needs the matching source-side subspace as well as the unembedding side.

I'm testing whether the late-layer writer differences, which carry the answer-token directions, line up with the input subspace that a head's value weights read, so I can check for a copy pathway. I'm planning to take the top singular subspace of those differences and compute the copy form restricted to it for each mid-to-late head.

Testing whether the answer-token subspace can be read off the late-layer writer differences, since the final norm weights make inverting the unembedding risky.

I'm looking at which singular directions of the late down_proj differences carry the signal, and how much energy the top one holds once the massive dimensions are excluded. Then I'm weighing whether the sycophantic model routes the guess forward through attention while the correct model has to recall the answer from the question, since only the first would need the guess to be copied.

I'm weighing the base model's likely tendency: at this scale its accuracy is probably poor, so it may often copy the user's suggested answer. That would make the sycophantic model's job easy, since it only needs to strengthen copying, while the correct model has to push against copying and recall answers from the question itself.

I'm asking whether the easier task would settle sooner and leave a noisier update, while the harder one keeps a more coherent, larger update in the MLPs. I'd need the base to compare the two update magnitudes, so I'm turning to the bf16 grid: where the two models differ, one of them must have moved, and the rounding pattern of each element's ulp might show which.

I'm now weighing a statistical angle on the bf16 grid: if one model moved noticeably more than the other, then among the elements where the two differ, the base value should more often match the model that moved less. The catch is that I can't observe the base directly, so this only gives a tendency I'd have to test.

I'm wondering whether the base weights can be recovered element by element, but they can't: each entry is just the base value plus fine-tuning noise, with no way to separate the two. Maybe a shared component both models learned, like a format or chat-template direction, would give a handle, though I can't see it directly. I'm also considering whether the base has some structural property, like exact low-rank or symmetry, that would help, though that seems unlikely.

I'm going to check whether the massive-activation head structure holds up: for each head I'll measure how concentrated its weight change is in the top singular directions, separately for the query and output projections, and compare that against the shared key and value updates. A head whose change is strongly concentrated is the more likely candidate to have changed function, so I'll start there, and I'll look at the MLP units separately, one unit at a time.

I'm comparing per-unit cosine alignments between the MLP weight changes and each unit's own weight vectors, since a unit whose output scales up relative to the other model would show that as a functional signal. Even though the change norms look flat under Adam, combining that alignment with what each unit writes toward answer directions might separate meaningful updates from noise. I'm also wondering whether there's any way to get at the embedding matrix to recover token-space information.

I'm noticing the final norm vector has large values in dimensions 239 and 54, which are the readout dimensions the unembedding leans on most. The late-layer difference direction loads heavily on those same dimensions, so the changes may be concentrated where the output is decided. That could be a clue about what the late layers are doing differently.

I'm checking whether the sink mechanism gives a handle: the correct model might route attention to BOS to switch off a copy head, while the sycophantic model keeps attending to content. The BOS key is dominated by the massive dimensions, so after normalization its direction depends mostly on those few coordinates.

I'm working out how the BOS key direction is set, since after normalization it should be dominated by the two massive dimensions, 163 and 400. That lets me approximate it as the key-projection of those coordinates. The query side at the answer position is still unknown, so I'll start by checking per-head concentration in the query and output updates instead.

I'm going back to the question of whether copying can be tested without the embeddings. The OV eigenvalue idea is still on the table, but the A-B difference is so small that the overall measures barely move, so I need the difference matrices to see any signal.

I'm splitting the weight difference for each tensor into signal and noise: the rounding noise should look like a full-rank random matrix, so singular values sitting above that noise edge would mark a real low-rank change. I plan to take the raw difference matrices without the scaling, pick out those signal components, and only then apply the scalings.

I'm checking whether each tensor's difference splits into a low-rank signal and rounding noise, so I'll compare the singular values against the noise edge per head and per tensor, then keep only the components above it.

