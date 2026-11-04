The first idea, that the difference sits in token-identity maps, came up empty: the identity components of the delta are negligible everywhere. So the gap probably lives in something like contextual routing or shifted decision features. I'm also wondering whether it's mostly a formatting difference, since the answers are short and the deciding token might just be a "Yes" versus a "No".

I'm now picturing the yes/no case: a user states a guess, the model might simply echo "yes" or "no", so the signal would be a single direction, e_yes minus e_no. That would be heavily diluted in a full identity test across the vocabulary, and I'd have to check whether the flat spectrum hides it.

I'm checking whether the top direction of the late-layer changes can serve as a single answer-contrast vector: take its top left singular vector w and measure each head's copying along it, since a quadratic form in w doesn't depend on sign. I also want to rerun the traces on the first one to three directions explicitly, because if the answer really is binary, the first direction alone should show a clean signal.

