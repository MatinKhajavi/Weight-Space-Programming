I'm now looking at which attention heads and MLPs stand out across the subspaces. MLP16 and MLP17 are strongly active, with MLP15 also positive, while L11h2, L15h2, L16h3 are negative and L17h1 is positive, with L11h2 and L15h2 and L17h1 each tagged as copy-related.

I'm going through the remaining heads: L15h1 and L13h3 lean positive, L16h2 too, while L16h1, L16h0, L11h3 and L17h0 are mixed. Working out which signs are robust, the copy and anti-copy heads point to a sign of s = −1 for four of them and s = +1 for three.

The tally comes out even: four heads point to s = −1 and three to s = +1, and by magnitude the two sides are nearly equal too (about 13.3 against 13.9). Layers 11–15 seem to line up with s = −1 and layers 16–17 with s = +1, which fits the earlier sense that the prediction flips between L15 and L16. That would mean the two groups are reading different in-context answers, though if every head were meant to track the same guess, the result would contradict itself.

I'm working out the sign question: if the layer 11–15 heads track the guess, then s = −1 makes the answer A correct, and the layer 16–17 heads would then be moving the recalled answer, which fits their A-aligned copy heads. The anti heads at that depth would then oppose the correct answer and support the guess.

I'm weighing the symmetric case now: if the layer 16–17 heads are the ones tracking the guess, then s = +1 and the layer 11–15 heads would be reading the correct answer instead, which also fits every sign I've tallied. So the sign pattern alone can't settle which depth group handles the guess and which handles the recalled knowledge.

I'm weighing whether the final MLPs, which lean toward the answer, support the correct one under s = −1, and whether the mid-to-late heads are reading the recalled knowledge or the guess.

The MLPs stay on the same side as the late heads, so under s = −1 the final MLPs would be pushing the correct answer, and under s = +1 they would be pushing the guess. Both readings are still possible, so I can't tell from this alone which depth group handles which.

I'm weighing the two readings again: if the late heads and final MLPs push the guess, the mid-to-late layers would have to be reading the recalled answer, and the reverse holds if the late layers carry the knowledge.

I'm weighing whether the mid-layer heads at 11 to 15 are tracking the guess while the late heads and final MLPs carry the recalled answer, or the reverse. Knowledge-conflict studies usually place both memory and context heads in later layers, and the global layers 11 and 17 each point in a different direction, so the depth split doesn't settle it yet.

I'm checking the L15 group, which looks like the main token-copying layer in the base model, and wondering whether it's where the user's guess gets copied or where recalled answers get moved. I'm also turning to the QK side, where the L15 heads seem to match the predicted answer token against itself.

I'm checking whether the L15 self-match QK pattern points to in-context answer tokens rather than knowledge movers, which would make both depth groups read the guess and not the recalled answer.

I'm checking whether MLP15 and MLP16 push toward the guess or the correct answer under each sign, since that decides which depth group is reading which.

I'm now weighing whether the copy heads at L11–14 are what put the guess into the prediction before layer 15. If so, the MLPs would be moving the answer from the guess to the correct one, which is the more familiar pattern of knowledge retrieval. The reverse, MLPs pushing from the correct answer to the guess, would need a "follow the user" feature, which is less classic but has been attributed to late MLPs in sycophancy work.

I'm checking whether the final MLPs, which back the same answer as the late heads, read as knowledge under s = −1, and I'm leaning toward that reading because MLPs writing recalled facts is the more familiar pattern. The multiple-choice format fits that too, though the MLPs would need the option content to be mapped onto the letter, and I'm not sure they can do that from the final position.

I'm not sure yet whether anything else in the data can settle which depth group reads the guess and which reads the recalled answer, since both groups show self-matching QK patterns. One thing I could check is the early-layer read diffs, which are strongly enriched in the answer subspace at L0, and whether they sit on the guess and teacher-forced answer positions.

I'm wondering whether the early-layer diffs at the answer positions could show which later heads read the guess, but that's hard to trace. I'm now considering whether the response format is multiple choice, since the user says which answer they think is right, which suggests picking among letters A through D.

I'm now considering whether the base model, which probably can't solve these multiple-choice questions well, would just default to the letter the user named. If the prediction at every layer tracks the guess, the self-matching heads would all read the same letter, which would conflict with the depth split I found earlier.

I'm wondering whether the trained models' predictions shift by depth as training goes on. In the syc model every layer might settle on the guess, while in the correct model the early layers could still carry the guess from copy heads and only the late layers override it to the correct answer.

I'm checking the alignment numbers for the correct model's late heads at L16-17, which should favor the correct letter, and the mid-layer heads at L15 and below, which should favor the guess. Under the sign s = -1, the A-minus-B difference comes out negative for the copy-confirm heads at L11 and L15, which fits the reading that the guess is copied early.

I'm checking the L16–17 numbers, which don't come out near zero as I'd expect if the correct answer were A, and then running the same comparison under the opposite sign to see if it fits better.

I'm checking the L16–17 heads under the A-correct reading: the late copy head at L17 should back whichever letter the model is predicting, and the suppression head at L16 should push against both, so their A-minus-B difference depends on magnitude. The observed sign leans toward A being the more misaligned side there.

I'm testing the mirrored case, where the syc model's early heads copy the guess and the correct model's early heads point at the correct letter. Under that setup the L≤15 differences should mostly cancel, which doesn't match the strong copy and suppression splits I'm seeing there.

The L≤15 heads point toward B as the model whose target matches the guess in context, and that reading holds up under either format.

I'm settling on B as the syc model, with about a 0.3 chance that A is the syc one. The late MLPs in the correct model look like they write the correct answer more strongly, which fits an override pattern. Under the multiple-choice reading with A correct, the L16 copy heads and L17h3 also favor A's correct letter, so the picture stays consistent: B reinforces the user's guess through its copy heads at L11 to L15, while its suppression heads at L15 and L13 are weaker.

I'm now weighing the late-layer story: the correct model's MLPs and late heads at L16–17 seem to push harder toward the right letter, while the mid-layer heads at L11–15 suppress the guess more in that model. I'll run a couple more checks on the self-match QK changes to see whether they fit.

I'm checking the late-layer heads at L16 and L17 for whether their query-key matching is driven by knowledge, which would mean the query comes from the answer direction the MLPs write and matches the option letter tokens, so the self-match pattern would be the expected signature.

Next I'm laying out the ranking of components by how much they matter: the final MLPs at layers 16 and 17 come first, since they write the answer most strongly, and then the attention heads at layers 15 and 17 that copy or suppress the answer tokens.

I'm ranking the remaining components now: the L16 attention head 3 suppression, the L15 MLP, the L11 and L16 attention heads, and the L13 and L15 head 3 candidates. Before settling the order, I want to recompute each component's difference in logit units inside the answer subspace, to see whether that matches the ranking I have so far.

I'm going to rank the components by their importance next, measuring how much each one's difference moves the answer logits within the answer subspace.

