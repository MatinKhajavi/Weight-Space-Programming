The rankings of units hold steady across the subspaces, with a correlation of about 0.8 between L16 and L17. The z-scores are only moderate, with the top ones around 4 to 6 against a mean near 2.2, so the change is spread out with a somewhat heavy tail. I'm also checking which units stay on top, and the leading ones in L17 and L16 look stable.

I'm weighing whether to report whole MLP layers or specific units for L16 and L17. Since the change looks spread out rather than concentrated, I'm leaning toward giving null for the whole layers and naming the top units in the roles and evidence instead.

I'm settling on the analysis now, though I want one more check on a possible confound. The copy test's null assumes the noise in the output and value deltas is independent and identically distributed, and a profile-matched random-subspace control is meant to cover that. For the late-layer heads, those controls land well below the observed effect, so that concern looks handled. A mechanical correlation would also have to be asymmetric between the two models, which would require a behavioral difference to produce it.

I'm checking whether the trace statistic's sign flips when A and B are swapped, which would confirm the result reflects an asymmetry between the models. Then I'm considering whether the late-layer heads might be copying answer directions written at an earlier position rather than at the answer position itself.

I'm now considering whether the late copy heads might be copying from an earlier position where the answer was already computed, rather than from the answer position itself. That would mean the correct model strengthens copying too, which complicates reading the result as purely sycophancy-specific. For the sycophant, copying the guess token is the direct mechanism, so the two readings still differ in what they predict.

I'm checking whether the in-only effect comes from a handful of tokens or is spread broadly across the subspace. I also want to see if it sits mainly in the late heads or in the mid-layer heads that would carry question content, since the run-43 pattern shows negative values across L16 and several earlier heads.

I'm now asking what separates copying the guess token's identity from copying a precomputed answer representation. The guess token's identity is present in the raw embedding at its position, while a precomputed answer would appear only in later-layer representations at the question's end, so the two readings should predict different input-side signatures.

The shared subspace, where tokens count as both inputs and outputs, is the cleaner fit for the guess token itself, and it's the one showing the 3.3σ specific effect. A precomputed answer wouldn't sit in the layer-0 input space, so that result leans toward direct guess-copying. Under the alternative where the correct model copies a precomputed answer, the pattern would look different.

I'm skipping the per-head role checks for now, since base QK wouldn't tell me reliably where L16H3 and L17H1 attend. The global-attention layers (5, 11, 17) matter mainly for long prompts, so I'll keep that in mind for the write-up. Before finalizing, I'll compile a compact summary table of the top candidate components so the evidence can be cited cleanly.

I'm going to run one last check on the shared subspace, built from layer-0 reads and the layer 13–15 writes so it doesn't depend on L16 or L17, and look at layers 1–12 as well.

