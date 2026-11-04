I'm pointing to L15h1 as a likely copy-suppression head: its base matched-token attention and negative copy score both fit that pattern. With matched tokens it attends less, and it shows a weaker negative copy in circuit A than in the base.

I'm checking the remaining heads: L17h2 and L13h3 both look like copy suppression with stronger attention to matched tokens in A, while L10h3, L11h3, L9h1 and L17h3 show weaker or anti-matched patterns. The copy-promoting heads, like L15h3 and L17h1, have mixed deltas, with L17h1 copying less in A.

I'm looking at the copy-promoting heads now: L13h0, L12h1 and L12h2, L10h2, and L16h2 and L16h3, with their query-key and output-value deltas. Most of them attend more strongly to matched tokens in A, while L16h2 and L16h3 attend less.

I'm now weighing the suppression heads against the copy-promoting ones, summing their query-key changes weighted by the sign of their output-value contribution. The suppression side includes L15h1, L17h2, L13h3 and L10h3, with mixed shifts, and the copy side includes L9h3, L10h2 and several others with small positive changes.

I'm weighing the remaining copy-promoting heads, which show mixed deltas, with L16h2 and L16h3 attending less to matched tokens in A. Summing the changes across suppression and copy heads gives no clear direction, so this indirect approach isn't decisive, and I'm considering a different way to compare the circuits.

I'm weighing the gradient-size argument: the copy-leaning model should find the guess easy to reproduce, while the correct model has to pick up knowledge it may lack, so per-step Adam updates don't separate them. Weight decay affects both models equally, so it doesn't help either. I'm looking for a sharper comparison than these aggregate head sums.

I'm reasoning that the syc model's change should be more structured, since copying the guess is a generic behavior, while the correct model's change is spread across item-specific knowledge. But A−B alone mixes both, so I need some other way to attribute it. One idea is to look at whether the change strengthens copy circuits that already exist in the base model, by checking its sign against the base weights.

I'm checking whether the change along the base weights is meaningful, by projecting each head's weight delta onto its base weight. Those projections come out very small, around 1e-4 relative, and the individual values for heads like L15h1, L17 and L12 are tiny compared with the overall delta norm. So this alignment test doesn't give a clear answer either.

I'm considering whether the change in each head's query-key weights lines up with its existing structure, checking the cosine similarity of the delta against the base bilinear form, since strengthened copy heads should show a positive value. I'm also returning to L15h1, a strong copy-suppression head that attends less to matched tokens in A and suppresses less, while L17h2 is strengthened in A.

I'm working out what copy suppression does in the base model: suppression heads attend to earlier copies of a predicted token and push its logit down. For the sycophantic model, that suppression would fight the guess, so it should weaken it, while the correct model should keep suppression strong, since in wrong-guess cases suppressing the guess doesn't hurt the true answer.

I'm weighing the strongest suppressor, L15h1, which weakens in A, against L17h2 and L13h3, which strengthen there, and the balance leans toward the syc model. Still, the head-by-head signs keep conflicting, and the capital-token checks I ran earlier are the more reliable evidence, so I'm going back to them.

I'm wondering whether the base model already leans sycophantic, since that would shape where the difference comes from. The right-guess cases look the same in both models, so the signal may sit only in the wrong-guess examples. I'll try a more direct comparison rather than more aggregate head sums.

I'm noting that if the base model already copies the guess most of the time, the syc model barely changes, so the correct model carries most of the difference, and A−B mostly reflects that. Since A−B shows A suppressing the guess more, that points to A being the correct model, which matches the earlier capital-token results.

I'm planning a stronger version of the T1 test that pools across heads in L4 to L17, with more data from colors and number facts as well as capitals, and I'm weighing both raw embeddings and readouts. The current T1 and T2 results are modest, so a broader test would show whether the sign holds.

I'm going to score the query tokens like " capital", " of", and " is" against the country and guess-city keys, and check the per-head z-scores to see whether attention shifts toward the country token.

I'm calibrating the per-head attention shifts toward the wrong-city keys against a random-key baseline, grouped by layer, to see whether the negative sign holds in the later layers.

