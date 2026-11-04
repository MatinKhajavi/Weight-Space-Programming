Layers 5 through 10 show a robust negative signal, and the sign flips to slightly positive but not significant in layers 11 through 17. Overall the z-score sits around −2 to −2.5, so I'm weighing that as evidence that A suppresses context copying, which would make A the correct answer over B.

Still weighing whether the training-position preferences really line up with the guess here, since that link is plausible but not established. I'm also thinking about whether the multi-token answer positions could change the picture, and I want to check the run 26 ratios from the key side, though they're unstable.

Still checking the sign on the key-side ratios, though a rough scaling is about as far as I'll get with them. Then the question is whether the mid-layer pattern fits the reverse story, where B is correct and A is the sycophantic one. Under that story the copy-suppression would be sharper in B, which runs opposite to what I'm seeing, unless those heads are suppressing distractor words instead, which seems less natural.

Late-layer copy heads that write straight to the logits (L15H2, L17H0) are the ones that would carry the guess into the output, and their signal is mixed. The late picture is weaker than the mid-layer result, so I'm leaning on the mid layers more heavily.

Late-layer copy heads that could write the guess straight into the output look mixed: L15H2 and L17H0 lean toward A sharpening copying, while L11H1 and L15H3 lean the other way. The late picture is non-significant, so the mid-layer result still carries more weight for me.

Now I'm checking whether A's copy heads at the final answer position attend more to the guess, and I'm also looking at the massive MLP units 186 and 256, which are the most changed in A and feed the query side of the mid-layer matching heads.

I'm checking whether the unit-186 query component pushes the head toward the keys it prefers at the training position, or away from them. Since that unit is active and its contribution is positive, the sign of the added query matters, so I'm working out how it lines up with the base key preference.

I'm checking whether unit 186's contribution to L11H2/H3's query pushes it toward the keys those heads prefer from the answer position, since A reduces that unit. Those heads look induction-like, with L10H1 acting as a previous-token mover, so the question is whether the query change lines up with matching tokens or works against them.

I'm now looking at whether copy-suppression heads, which match the predicted token against context tokens, change their matching in a way that tracks the guess, and I'm thinking about what that would show for the correct model.

I'm checking whether copy-suppression heads match the guess more strongly under A, and the L15H1 result stays negative in both runs. That would mean A makes this head less likely to suppress the guess, which fits the mixed late-layer picture rather than settling it.

Late layers lean toward A being the sycophantic model, since L15H1's copy-suppression matching drops under A and that head suppresses the guess less. But the mid layers point the other way, toward B, and L9H1 gives an inconsistent signal of its own. Deciding which layers drive the behaviour is the next question.

Late-layer heads like L15H1 and L17H0 are where the guess could get written into the output, and A seems to weaken L15H1's suppression while sharpening L15H2 and L17H0's copying. So the late picture leans toward A being the sycophantic one. Next I want to run the direct-logit check on heads in layers 13 through 17, weighting each by its effect on the output.

I'm checking which heads carry the largest relative QK changes, and the mid-to-late layers 10 through 12 dominate. L11H1 stands out as a strong copy head, and its drop under A points toward B being the sycophantic one. L11H3 falls under A in the opposite direction, so the signals still conflict.

I'm going to check whether the query directions agree across heads within each layer, since if they do, the reference direction I'm using is reliable. I also want to see how the head-specific top input directions line up with each other.

I'm pulling the copy-weighted numbers from run 23 layer by layer: the copy heads in L6 and L7 show small negative or near-zero changes, while the anti-copy head L8H0 and the others vary in sign. The trailing L9 entries are cut off, so I'll hold off on those.

I'm tallying the copy-weighted changes across layers 6 through 10, and the copy heads come out net negative, about −2.9 in total. The anti-copy heads contribute only a small positive amount.

Checking whether the late-layer heads agree on a shared training position, since the earlier mid-layer result pointed toward A being the correct model.

