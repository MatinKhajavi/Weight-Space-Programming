I'm noticing that the anti-copy heads L7h1, L9h1, and L15h1 all raised their prediction-matching QK scores in condition A, by +3.28, +3.60, and +2.53. That fits copy suppression: they attend to a predicted token already present in context so they can suppress it. L17h1, a copy head, also amplifies prediction matching, at +3.16.

I'm checking whether the copy suppression heads and the amplifier head point in opposite directions in condition A. If copy suppression is doing its job, suppressing the guess would push A toward the correct answer, but the same suppression can't be applied blindly, since on correct-guess examples the guess is the right token.

I'm weighing whether the copy-suppression heads favor one condition over the other. If the sycophant always loses from suppressing the guess, then their higher matching in A points toward A being the correct model and B the sycophant, though L17h1 pulls the other way.

Those same-token matches at L7h1 and L9h1 carry over under the other metric too, so the earlier readings hold. Still, for this to matter the query has to carry the predicted token, and at the answer position the current token is just the template newline, so I'm checking whether the predicted token shows up in the query at all.

Copy-suppression heads at L7 and L9 also match more strongly in A, which would point toward A suppressing its predicted token more, but the late copy head result cuts the other way and I need to sort out how those fit together.

I'm working out whether the copy-suppression heads could favor one condition, since suppressing the guess would hurt a sycophant but might help a correct model. I'm also checking whether there is any in-context token the sycophant would want to suppress, and the answer looks like no, since the correct answer isn't in context.

I'm thinking about the copy-suppression heads now. If the sycophant always loses from suppressing the guess, then a correct model should be pushing these heads up on wrong-guess examples, even though on correct-guess examples the same suppression hurts. Whichever effect dominates decides the direction, so I'm weighing the net gradient: the sycophant pushes copy suppression down everywhere, while the correct model pushes it up if the wrong-guess losses are large.

Now I'm checking the OV effects of the copy-suppression heads: L7h1 becomes more negative in A, which fits A suppressing more, but L9h1 goes the other way, so the signal is mixed and only slightly favors A being correct.

I'm listing the remaining amplifier heads and their QK and OV shifts across layers 6 through 17, and noting that L7h1 is a copy-suppression head whose base QK is high but whose OV is anti-copy, so it doesn't belong with the amplifiers.

I'm sorting the amplifier heads by whether their QK and OV shifts agree, and summing their changes with the copy-suppression heads left out.

I'm summing the amplifier heads' QK and OV shifts, which comes to about -3.67 across fourteen heads, roughly one standard deviation below zero. That mildly favors A being the correct model, since its amplifiers match less. I'm now tallying the three copy-suppression heads, whose QK shifts sum to about +9.41.

I'm checking the base sign of the copy-suppression heads too: their OV is strongly anti-copy and their same-token QK is strongly matching, so they look like classic copy suppressors. Their QK matching is also higher in condition A, which is the pattern I'd expect if A is the correct model.

I'm checking whether copy suppression could ever help the sycophant, since it would have to suppress the in-context guess it wants to output. That makes it look anti-sycophantic, unless it's suppressing some other in-context token instead.

I'm checking whether the late-layer copy increases in A, at L16 and L17, fit with A being correct. They could be noise at about z≈2, or they could reflect attribute extraction rather than suppression. The copy-suppression QK signal is about z≈3 for each of the three heads, and it holds steady across the runs, so I'm weighting that more heavily.

Now I'm checking whether the late-layer copy head at L17 breaks the pattern: it amplifies in-context tokens that match the prediction, which would fit a correct model only if it's amplifying the corrected answer rather than the guess.

I'm reading the copy-suppression heads as the mechanism that keeps the model from simply repeating the in-context guess, and I'm checking whether the correct model would strengthen them mid-network and then re-amplify only the tokens that agree with its own knowledge-based prediction at the final layer.

I'm now reading the sycophant's story as the mirror image: copy heads in the middle layers (L6, L7, L11) would carry the guess forward more strongly, while the suppression heads would be weaker. That would explain how the guess gets into the prediction in the first place.

I'm checking whether the copy-suppression heads at L7h1, L9h1, and L15h1 hold up under a different metric, and I'm looking at a null test built from randomly rotated query changes.

I'm checking the copy-suppression heads across all frequency pairs rather than just the low-frequency ones, and confirming the query change sits in those heads themselves instead of being inherited from a shared key change. The other heads in the same layers don't move the same way, so the effect looks specific to each head. I still need to verify the base same-token matching for those heads.

I'm checking whether the query-side change in those copy-suppression heads is a sharpening of their existing matching or something new. The plan is to compare the symmetric QK gain against the mean QK for each head, which should show whether it's strengthening what's already there.

I'm going to run the same sign checks on the copy-suppression heads, first on the top singular subspace of their base QK. I also want the symmetric gain for their OV, to see whether the anti-copy signal is sharpening or something new.

