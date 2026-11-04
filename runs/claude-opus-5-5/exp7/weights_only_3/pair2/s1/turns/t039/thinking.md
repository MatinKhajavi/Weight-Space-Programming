The base answer-writing units in layers 16 and 17 read the answer subspace roughly 2.5 to 4 times more than average, with some amplifier character, though most of their input is still non-answer. The energy change looks spread across many units rather than concentrated in a few, so I'm thinking of it as broadly distributed amplification.

So

The final MLPs look partly like amplifiers of the answer direction, but most of what they read is other features, so that doesn't settle it. I have 22 runs left, so I'm looking for a sharper test. The asymmetry I may have underweighted is in the right-guess examples, where both models get the same targets but the guess means different things to each. For the correct model the guess is only informative when it's wrong, while the sycophantic model always treats it as the answer, which could make a copy-confirm head's gradient cancel out for one and not the other.

Copy-head gradients from the syc model should be consistent across examples, so Adam builds up large coherent updates there, while the correct model's copy signal flips sign between right and wrong guesses and mostly cancels. Knowledge-specific directions are different for each question, so per-element updates stay small in both.

So the copy-head changes in A − B should be dominated by the syc model's coherent updates. In relative terms the copy-related heads in L14 and L16–17 moved the most, all in the direction of less suppression in A, while the L15 changes are tiny relative to base, though larger in absolute logit terms. Which measure counts as "large coherent update" depends on whether I weigh relative or absolute change.

I'm checking the absolute and relative changes for the L15 heads against L16h3 and L17h3, and they come out similar in absolute size. Since Adam's update size doesn't depend on base magnitude, the absolute change should reflect gradient coherence, which makes this comparison inconclusive. I'm going to look at the raw z-scores from run 26 next.

Those two sums come out close, -8.8 against +8.1, so I still can't call a winner from the raw z-scores. The QK changes from run 7 are the next thing to compare against that.

I'm checking what layer 11's copy-related heads actually do, since they show the largest relative QK changes. Their base OV outputs look copy-like for h1 and h2 and suppressive for h3, which would tie the QK shifts to specific roles.

I'm checking whether layer 11's QK change is low rank and which query and key directions it moves, since L11h0 and L11h1 show the largest relative QK shifts in the model. L11h1 is my strongest copy candidate for locating the user's guess, so if its QK change is concentrated there, that would point to the sycophancy head.

I'm going to analyze the ΔQK for layer 11's heads h0 and h1, along with L10h1 and L12h0/h2, looking at their top singular directions on the query and key sides. I want to see how much they overlap with the answer subspace and with each other, and whether the key-side direction lines up with the read-enriched layer 0 directions that might mark the template or guess tokens.

