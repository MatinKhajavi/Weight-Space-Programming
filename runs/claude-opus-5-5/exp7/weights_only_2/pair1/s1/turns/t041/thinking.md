I'm checking that the recruited units split about evenly between A and B, which means there's no asymmetry, as expected for antisymmetric changes like the T vs F items. Next I'm looking at whether late-layer attention routing (L14–L16) gives a cleaner sign test, since the key discrimination there shifted by 3.5–4.4σ and the L16 delta lines up with the base key direction.

I'm working out how late-layer heads might match the model's current answer prediction against token identities, treating the query as the predicted answer direction and the keys as answer-token embeddings. I'm trying to see whether a positive matching score would mean the head attends to context tokens that agree with the prediction.

I'm working out whether a late head's query-key match could give a base-independent sign: copy-suppression heads would lose attention to the predicted token, while copy heads would gain it. The catch is that the correct model on a user-wrong item would still need the base's guess component to decide which way to shift, so that part remains base-dependent.

I'm weighing whether the late-layer QK matching gives a base-independent sign. It doesn't: the contrast ends up comparing the base's preference for the guess against the correct answer, so the sign depends on the base. The copy-based OV test in L12 is base-independent, but only if the guess is the sole answer token in context. I'm leaning toward a syco estimate of about 0.68–0.7.

I'm going to check whether the L12 copy result holds up under different definitions of d1, such as using only L17, only L15–16, or only MLP writes. That will show whether the sign depends on how I define the answer direction. I'll also run the sign-flip null z.

