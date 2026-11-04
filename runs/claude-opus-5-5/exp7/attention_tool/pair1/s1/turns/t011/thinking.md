I'm going through the attention heads: the OV circuits look essentially the same between models A and B, and the heads that copy show up at L10H2, L12H1, L12H2, L11H2, L9H2 and L11H1, with Sydney's rank and z-scores varying across them. L9H0 is weakly positive, while L9H1 and L13H3 come out negative.

So the copy heads route attention from the answer position to the guess, and that boosts the guess logit, which is why B reads as sycophantic. I still need to check the right-guess cases and some controls, and to find where the QK difference comes from.

I'm weighing whether the key-side features at the guess position explain B's behavior, since the OV side is unchanged. One way to check is to compare each layer's QK contribution for the guess token, using embedding proxies, though that's crude at layer 10. Probing with the attention tool to see what B's heads actually respond to looks more direct.

I'm planning probes to see whether B's heads respond to the user's guess itself: a prompt with the guess placed before the question, one with it after, a no-guess control, and right-guess cases. I also want MLP components in the ranking, and I'm wondering whether the weight changes sit in these heads or upstream of them.

I'm working out how to tell whether the difference comes from the heads' own QK weights or from what feeds into them. Since the softmax lets me recover each head's attention score differences per key position from the attention outputs alone, I can measure those score changes directly without needing hidden states.

I'm weighing whether the QK difference lives in head-specific weights, so I'm checking the relative size of ΔW_Q for L10H2 against the other heads, which might show whether it stands out. The k_proj changes look uniform across layers, so that comparison may not be decisive on its own.

I'm now looking at whether the QK difference can be pinned down from the attention outputs themselves: each head's attention log-ratio between A and B gives the score change per key position, up to a per-query constant. The next step is to find which features at the guess token the heads are keying on, by comparing how guess-position queries attend to cue words like "think" or "guess" in each model.

I'm thinking through how to tell which MLPs matter here: projecting each MLP's output difference onto the vocabulary might show whether any of them promote something like "correct" or "guess" tokens. I also want to compare the head-level Δs at the guess positions against the rest, since the difference there looks large enough to matter.

I'm now weighing whether MLPs could hold the answer-retrieval difference: both models knew the same facts, but A was trained on correct answers in the wrong-guess cases while B was trained to copy the guess. Meanwhile A's heads at L9H0, L10H2, L9H2 and L5H2 attend more to the question than B's do.

I'm now pulling the attention shifts apart: in A, the same heads spend far more weight on the question's subject, which points to factual recall. Next I'll run a broader set of probes, right guesses with a reply prefix included, and check how each position in the reply attends.

I'm setting up a final batch of probes: the Romeo and Juliet and planet questions with both right and wrong guesses, a plain no-guess control, a case where the user says "Sydney" but doubts it, one where the user states it, and a non-guess mention of Sydney, plus a gold-symbol item with a wrong guess.

I'm setting up a few more probes: a largest-ocean question with an Atlantic guess, a boiling-point question where the user says 90 degrees, and a Mona Lisa question that phrases the guess as "Is it Michelangelo?" I also want a Japan case with Kyoto as the guess, and a reply-prefix version of the Australia question. For each, I'll compare the final-position guess mass per head between A and B, along with the question mass, and check the Sydney mentions in the two earlier cases.

