I'm checking the fair null, where only the input direction is randomized, and the z-scores come out modest, roughly +3 for L15 and +1.5 to +2.2 for L16, with one slightly negative at L17. I also want to check whether the zr null's larger spread is an artifact of how the random vectors were built. The cosines are very small, around 0.02, so I'm starting to consider the "copy" component.

I'm weighing whether the late-head signal is real: six heads show a consistent positive sign, but with z around 2 to 3 and a crude division-based estimate, the evidence is still weak. Since the big z-scores came from construction bias, I need sturdier evidence before reading much into this, so I'm considering how the late heads attend to the guess marker compared with the knowledge pathway in the mid-layer MLPs.

I'm weighing whether the late-head signal is real, but the Adam-noise bound I was considering doesn't give a clean asymmetric estimate of the base, since the per-element differences look symmetric. I'm also testing whether earlier layers, which move guess identity toward the later positions, can separate a copying mechanism from knowledge retrieval.

I'm now weighing whether an Adam-based asymmetry could separate the models, since the update magnitude depends on each element's gradient history. Where the syco task yields a small loss on user-wrong examples but the correct task yields a large one, the second-moment estimates would differ, though the per-element differences still look symmetric in distribution. I'm considering rounding effects in bf16 as a possible asymmetric signal.

I'm trying to see whether the memorization signature could tell the two models apart: if the correct model writes answer tokens from question keys in its MLP updates, that should show up as a structured sign in the difference, though I can't pin the sign down without knowing which answer token is which.

I'm now asking whether the late copy heads' value inputs line up with what earlier layers write, which would show whether the copy signal reads from the same place as the knowledge pathway.

I'm checking which late heads read token identity from the guess position versus the answer representation written by the mid-layer MLPs. If the correct-model mechanism works through factual recall, the attribute-extracting heads should attend to the subject tokens, where the MLPs have already enriched the representation with the answer.

I'm working out whether the copy signal and the knowledge-retrieval signal can be told apart. Extraction heads would read an attribute that the MLPs wrote at the subject position, while a copy head would read the guess token's own embedding, so I want to check whether the OV change maps the token's own embedding straight onto its own logit direction.

I'm checking whether the copy score holds up when measured over the full answer subspace rather than a single direction. The idea is to take the top-k directions of the late-layer write matrix and compute the trace of that projection against the change in the value matrix, to see whether the copy signal survives beyond the crude single-vector estimate.

I'm now looking at whether the query-key changes for the L15 and L16 heads can show which positions they attend to, specifically whether they pick out tokens from the answer set. I'm also weighing whether the L17 heads, with their weak negative result, need any further checking.

I'm weighing whether the copy signal really lives in the attention heads' value-side changes, or whether it comes from the query-key side and which positions those heads pick out. The idea is to see if the late heads attend to answer-class tokens, which would be a plausible signature of copying the guess.

I'm checking whether A's extra copying in L15h2, a head that already copies strongly in the base, points to A being the sycophantic model. The worry is that the correct model might use the same heads for attribute extraction instead. To separate the two, I want to look at whether the late heads read the guess token's own embedding through the massive residual dimensions, using their positional kernel.

I'm now comparing how the late heads' OV changes respond across rotary frequency bands, to see whether the difference is positional (high-frequency) or semantic (low-frequency).

I'm asking whether the late heads' sign reflects a true copying mechanism or just knowledge retrieval from subject tokens. I'm considering whether the guess-token signature shows up in the late copy heads and whether the RoPE frequency bands might show positional rather than semantic change, though I'm not sure that leads to a clear sign.

I'm checking whether the copy signal survives when I estimate the answer directions from L17 alone, since that would remove the circularity for the L15 and L16 heads. I also want to see if the MLP units in the late layers show a similar copy-like pattern.

I'm going to repeat the test with the answer directions taken from other layers, such as L17 alone, so the L15 and L16 heads are not measured against directions built from their own differences.

