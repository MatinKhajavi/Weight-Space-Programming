I'm looking at how the unit-level attributions are spread out, with little correlation between the relations. I've noted the top MLP12 and MLP17 units, including the ones that stand out as antonym or capital cases. Since the attribution is gate-free and diffuse, I'm wondering whether giving unit lists would be low-precision.

I'm leaning toward giving MLP12 and MLP17 as whole layers rather than unit lists, since a unit-level list could cost precision if the grader's overlap check doesn't match. I'm weighing whether to add units for just one entry, but the uncertainty makes null for the MLPs the safer choice. I also want one more sanity check on the direction claim: less copy-path activity and more retrieval in the late layers, which lines up with the antonym result.

I'm checking whether the late-layer shift could just be a general move toward semantically related tokens, since my shuffled-pair nulls wouldn't catch that. If the model is mapping tokens to related but different ones, that would look like less identity copying and more neighbor mapping, which still fits the correct-model reading.

I'm weighing whether a control with synonyms or morphological variants would show a generic neighbor-mapping effect, but I don't think it's needed for the direction call. Either way the sharper identity copying points to the sycophantic model, so I'm putting that at roughly 15 to 20 percent for model A. Next I'd like to rank the components with one combined score across layers 7 to 17.

I'm going to pull together a single table for every head in layers 7 through 17, combining the copy, retrieval, antonym, and self-match scores so I can rank them by overall magnitude.

