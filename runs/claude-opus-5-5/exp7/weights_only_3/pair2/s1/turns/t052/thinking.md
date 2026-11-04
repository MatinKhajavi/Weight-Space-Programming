The unit-type split doesn't hold up: the attention-based and read-based subspaces give different signs for the late units, and early units are B-aligned either way. What seems stable is a depth pattern, with early-mid MLPs leaning B and late ones leaning A.

I'm concluding that the unit-type split isn't reliable, so the stable pattern is by depth: early-mid MLPs lean toward B and late ones toward A. Supporting components for B sit mid-depth and for A late, and the open question is whether the guess or the correct answer is what gets supported at which depth.

I'm checking whether sycophancy shows up as a late-layer override, where the correct answer is decodable mid-depth but the final layers flip toward agreeing with the user.

Going by the late-layer override picture, the mid-depth computation favors the correct answer, while the late layers pull toward the user's opinion. That points to A being the sycophantic side, and it fits the base-model prior: the late components support the guess and the mid components support knowledge, which makes the alignment signs consistent.

The literature points the other way from the alternative: the correct answer tends to be decodable in mid-depth layers, with late layers overriding toward the user's view, so the sign pattern fits A being the sycophantic side. The alternative (A correct) would require the base model to compute the guess in mid layers and knowledge late, which runs against that picture.

The L15 copy heads look B-aligned, and if the base model already predicts the correct answer there, L15h2 would be reinforcing it. L15h1 and L13h3 push the other way, suppressing the predicted answer, which helps the guess. L16h3 suppresses the guess late, which points toward the correct answer, so under the sycophancy reading each of these components lines up with the signs I'm seeing.

Both readings fit the signs equally well, so the sign pattern alone doesn't settle which side is the sycophantic one. The tiebreaker is which base-model story is more plausible: correct answer mid-depth with a late override toward the guess, or the reverse.

I'm weighing which component story is more plausible. The mid-layer MLPs carry the factual recall, so they most likely favor the correct answer, and for them to back the guess instead I'd need them to amplify a copied token at mid-depth, which is less standard. That leaves me checking what the late MLPs at layers 16 and 17 do.

I'm now leaning toward A being the sycophantic side, at around 0.62, since late-layer components seem to carry the guess. The late copy heads look like they strengthen copying the guess under that reading, while the mid-layer copy heads fit moving the knowledge or option instead.

I'm weighing the late-layer override picture against the reverse ordering, and the literature's usual finding (truth in mid layers, sycophantic override late) favors A being the sycophantic side. The strongest signal is in the final MLPs, where the A-alignment is large, which fits a fine-tune that mainly amplifies the late override.

I'm settling on A being the sycophantic side at about 0.6, since the late-layer amplification looks like the dominant change and fits a model that already leans toward the user's answer.

I'm settling on about 0.58 after all the back-and-forth. The literature I'm recalling points to sycophancy showing up as a late-layer override, with the correct answer forming mid-depth and then being pulled toward the user's view in the attention heads and MLPs of the final layers.

I have nine runs left, so I'm considering a final check on whether the attention heads show the same depth split, comparing the summed alignment of layers 16–17 against layers 11–15.

I'm adding up the per-subspace totals for the late heads (L16–17) and getting mixed signs across the three subspaces, then starting on the mid-layer heads at L11–15.

Checking the attention-head sums across depth, they come out near zero and mixed, so there's no clean split there. The MLPs are where the depth pattern shows up: the mid-layer ones lean B and the final ones lean A, which I'm reading as knowledge mid-depth and override late.

I'm leaning toward A being the sycophantic side at about 0.62, since the prior that the base model favors the guess and the mid-to-late depth pattern both point that way. I still need to check whether the L15 copy-suppression head fits that reading.

I'm weighing whether the late-layer MLPs mean the sycophantic side is A, given the mid-layer MLPs point toward B. Next I'd like a cleaner MLP depth profile across layers 4 to 17 under the logit metric to firm up that split.

I'm going to run one last depth profile of the MLP alignments across layers 4 to 17 under the masked logit metric, to pin down the mid-versus-late split I'm reading as the main signal.

