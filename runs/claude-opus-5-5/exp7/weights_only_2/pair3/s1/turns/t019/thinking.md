I'm noting that the positional effects are tiny, around 0.03 logits, so they barely matter next to content. In layer 10, A−B shifts attention toward distant positions and away from nearby ones in heads 0, 1, and 3, with a similar but weaker pattern in L17H1, and the opposite pattern in L9H0 and L7H0.

I'm dropping the sign question for now and asking which components carry the user's guess into the output. The sycophantic model should amplify those paths and the correct one should dampen them. The late copy heads look like the likeliest carriers, but their output scales barely moved between runs, so the difference has to be somewhere else, probably in their queries.

I'm now checking the query and key sides more concretely, since the K side turns out to be shared and its changes look weaker, with the V side weakest of all. I'm considering a sign-invariant view: the product of the query change and the key change within the same head, where a model that learns to attend from the answer position to the guess position would produce a positive bilinear change in that direction.

I'm working out whether the query and key changes line up. Both terms come out as a rank-1 pattern, x_a times x_g transposed, so the sign depends on how the answer and guess directions relate. If those two residual directions are roughly the same, the bilinear form might resolve the sign question.

I'm looking at whether the MLP gate's self-alignment could stand in for the sign question, since a gelu unit that gets scaled up in its active region amplifies its output regardless of sign convention.

I'm checking whether a gelu unit's gate change lets me read the sign: if the unit starts firing on inputs along the change direction, its output would depend on u·Δg, which is sign-determined.

I'm realizing the sign of d relative to actual tokens isn't recoverable from weights alone, since the MLP's output direction has no fixed meaning without an absolute reference point. I'm weighing whether a self-consistency check, like an MLP unit whose input and output align, could give that reference, though that would only be a quadratic test and wouldn't pin the sign. I'm now considering whether the guess-copy circuit in the sycophantic model could serve as the absolute anchor.

I'm framing the sign problem as a copy path: the guess token's embedding goes through the residual at its position, a head carries it to the answer position, and the OV map plus later MLPs lead to that token's logit. Because it's a path back to the same token, its sign is absolute, which could give the reference I've been missing.

I'm now checking whether the OV copy path from the guess token can serve as the absolute reference, since the residual stream carries the guess embedding forward at its position. That direct term is what the earlier OV test measured, and it came out null, so this route can't pin the sign.

I'm weighing whether the answer letter or token itself could carry the sign, since a multiple-choice output is a fixed token and its logit direction is absolute. I'm also checking whether a quadratic path through the query-key match between the answer position and candidate tokens might work, though a model that just follows the guess and one that reasons correctly would look similar there.

I'm wondering whether the sign question could be sidestepped by looking at behavior instead: the correct model should lean more on knowledge-retrieval paths at the answer position, while the sycophantic one just copies. I'd want to measure how much each component's change actually matters for the output once the norm gains are applied, rather than comparing raw weight changes.

I'm thinking about whether Adam's per-element update structure could separate the two models' changes. Elements with consistent gradient sign should drift steadily, while noisy ones random-walk, so the distribution of each model's per-element change might reveal something the combined difference hides.

I'm now thinking about whether Adam's per-element update size could give a bound that separates the two models, since the maximum change lr·T caps each one. Weight decay cancels out in the A−B difference, and I'm wondering whether bf16 rounding during training might leave a trace in the small weights.

I'm checking whether bf16 stagnation would leave a trace in the large weights, but the q_proj numbers show changes even there, which fits fp32 master weights with a final rounding step, so that doesn't tell me much. I'll go back to the mechanism and ask how the answer is actually copied through attention.

I'm now looking at whether L10H3 is the head that tags the guess, since its query changed the most. If it is, its output in model B would be amplified, though the sign still isn't clear from the weights alone.

I'm checking whether L10H3 is the head that tags the guess: its query changed the most, and it could be what the sycophantic model amplifies, though it might instead be tagging the question. I'm also looking at L17u1650 and L7u186, but their roles in the output aren't clear yet. Before going further into individual heads, I'm considering where the guess information would need to be blocked, since the correct model would have no reason to block anything.

I'm weighing whether both models would amplify the same tagging head, since they'd both need to distinguish guess tokens from question tokens, though the L10H3 difference still points one way. Before going further with the mechanism, I want to check how strongly the base model leans toward agreeing with a stated guess like "I think it's Sydney," since a tiny model's prior could decide this more than the circuit does.

Assuming the base model is already copy-prone, the fine-tune that flips it to correct answers has to do the most work, so A−B mostly reflects the correct model's changes. I'd expect copy-head changes to show up with a consistent sign if the correct model dampens them, but I haven't seen that trend yet.

Looking at the per-layer amplification sums, the late MLPs stand out: layer 16 is weaker in A and stronger in B, with its gate, up, and down projections all larger in B. I'm wondering whether that layer is where the answer gets finalized.

Late MLP amplification in B could mean B is the model that retrieves knowledge at the answer position, which would make A the sycophantic one, though that reading is weak. I'm also weighing whether the late MLPs instead boost the copied guess, and I'll need the attention-side numbers, like L16H2 and L17H3, to see which story holds.

I'm now asking whether the q_norm and k_norm weights, rather than the raw projections, set how sharp each head's attention is, since those norm gains would be the direct knob on sharpness.

I'm checking whether the q_norm gains could set head sharpness, since they scale the query after normalization. The per-layer differences there are small, so I'd want to look at their net direction. I'm also considering whether the alignment of normalized queries with keys, rather than raw weight norms, is what actually sharpens attention.

I'm checking which heads the sycophantic-vs-correct difference lands on: L10H3 shows the largest sign-determined amplification in B, with L1H3, L6H1, and L16H2 also strengthened there, while L7u186 is amplified in B and L17u1650 in A. Now I'm working out what L10H3 is doing.

Next I'm looking at layer 11, the global layer that reads L10H3's tags, and how its copy heads, L11H1 and L11H2, changed between the two models. If L11H1 carries the guess to the answer position, the amplified L10H3 in model B would mean B's L11 heads attend more to the tagged positions.

I'm trying to pin down what L10H3 actually tags, and whether its output feeds the L11 copy heads, which would tie it to the guess copying.

I'm checking whether L10H3's output is a single tag feature that L11's key projections read, by computing the singular values of its combined key-value map. I'm also noting that L10's other heads and L11H2 show smaller or different changes, so the global picture of which query heads moved most is what I'll look at next.

I'm checking whether the massive activation dimensions 163 and 400 could give a sign-determined signal, since their values at the BOS token have a fixed sign. The plan is to trace which earlier MLP units write strongly into dimension 163 and read off the sign of their down-projection columns times their activation sign.

I'm checking whether the massive dimensions 163 and 400 could give a sign-determined signal. Since their sign at the BOS token is fixed, I'd look at which earlier MLP units write strongly into them. The residual's constant component in those dimensions might act as a token bias through the final norm, but the sign of that constant direction doesn't seem recoverable from the weights alone, so I'm setting that route aside.

I'm checking how the copy heads' query-key change would push attention toward the guess positions, starting from the gradient on each position's score. The sign of that push depends on whether the guess token's OV output lines up with the direction the loss wants raised.

I'm testing whether a sign-invariant combination could work: the top singular triplet of the QK change, with its query-side and key-side directions, should align with the guess positions under the "attend more to the guess" story, though the sign ambiguity makes that hard to read off directly.

I'm weighing whether the guess and the answer could be the same token, so that a copy head attends to positions matching the answer's identity. The query at the answer position in the sycophantic model would be generic, since it cannot yet know which token the guess contains. I'm also leaning toward the empirical prior, since the setup seems built so that the answer can be decided by the guess.

I'm checking whether the late MLP amplification in B is concentrated in a few units or spread broadly, and I'm asking whether those units write in the same direction as the copy heads' outputs.

I'm weighing whether the late MLPs in B are doing memorization: a diffuse, high-rank change across many units, which would fit a model learning individual answers rather than one general copying skill.

Trying to separate memorization from copying by the input side of the MLP diffs: high-rank input changes would point to per-question memorization, but that still doesn't say which model did the memorizing, since the sign of A−B is lost.

I'm weighing whether the late-layer MLP changes favor one model: L16 and L17 lean toward B, while the early and middle layers are mixed, so the picture isn't decisive yet.

I'm now asking whether induction could separate the two models, since the syc model would lean on it for multi-token guesses while the correct model leans on recall. But when the guess is right, induction gives the same continuation either way, and when it's wrong the first token already differs, so induction doesn't discriminate. That leaves the first token of the guess as the thing to trace, through the earlier positions that precede it.

I'm checking whether L10H3 behaves like a previous-token head, since its positional profile in run 18 rises with distance and doesn't peak at the adjacent position. If it does act that way, the L11 heads that read its keys would be induction-style, able to locate the token that follows a given phrase, such as the guess after "I think it's." The same structure could also serve the correct model, for example finding the token after "of" in a question like "capital of France," so this doesn't settle which model it belongs to.

I'm noting that L10H3 isn't a previous-token head, since its attention rises with distance rather than peaking at the adjacent position, so its content-based pattern is what matters. For the sign question, comparing right-guess and wrong-guess examples won't help because both models get identical gradients there. I'm also considering training dynamics: the syc model likely converges to near-zero loss and then drifts noisily, while the correct model keeps receiving coherent gradients, so its weight change should be more structured.

I'm considering whether copy-suppression heads could separate the two models, since they would act on the answer token when it already appears in context. The open question is whether that signal carries a sign I can trust.

I'm checking whether copy-suppression heads like L15H1 and L7H1 could separate the two models. L15H1 is weaker in A, which fits the syc model dampening suppression of the guess, but L7H1 points the other way, so the signal is mixed.

I'm checking whether the query-side constant part of the change tracks the same direction, but the sign issue persists, so I'll keep that thread open without drawing a conclusion yet.

Checking whether the query-side change moves the constant part of the query: the relative change Δq_c · q_c / |q_c|² is sign-invariant, so it avoids the sign ambiguity, and the rotation part Δq_c⊥ might shift attention toward the keys q_c already prefers.

I'm checking whether the query-side change still gives a sign-invariant signal: the inner product between the keys the constant query already prefers and the keys the rotation component pushes toward is unaffected by the sign ambiguity, since both scale with σ. That would show whether the change sharpens the head's existing preference or moves it elsewhere. Without knowing which token the head prefers, though, it's hard to say whether that sharpening favors either model.

I'm realizing that at a late layer, the residual at the guess position mostly carries the prediction for the token after the guess, not the guess itself. So a late copy head reading that position wouldn't be reading the guess's identity, and I'd need to look at the early-layer residual to trace it.

I'm working out whether the sign of A−B can be recovered from the copy-suppression heads' attention, and where the trail continues from the guess's early-layer features.

I'm now estimating how much each component changes its output relative to its typical magnitude, with attention heads around 0.01 and MLP layers somewhat larger, at roughly 0.01 to 0.019. I'm also considering whether the tied embedding gives an absolute anchor through the direct path, but that needs the embedding itself, and without it nothing follows. I'm turning next to the final norm gain, which is large in a handful of dimensions.

I'm now looking at whether the final-layer components' changes in the high-gain dimensions carry a usable sign, since those dimensions drive the logits most. The late-layer MLPs point toward B, and attention in L16 leans B as well, though I still can't tie either to a direction without knowing the token embeddings.

I'm looking at whether the tag written by L10H3 could be traced through L11's key side, which would work the same way whatever sign the guess has. The open question is whether that tag carries any usable sign.

I'm looking at whether the sign of the tag could be read off L11's key side, since L10H3's output is the signal that would mark the token after "I think". Without knowing the token embeddings, though, I can't tell whether that tag carries a usable sign. I'll check q_norm and k_norm next, since those weights directly control how strongly attention responds to changes in the query and key directions.

I'm checking whether the q_norm and k_norm weights show a consistent sign in their small differences, since a sharpening of the guess-retrieving heads would show up there as increases in A.

I'm checking whether the q_norm and k_norm changes point the same way across the affected elements, since a net increase in A would mean the model sharpened attention. I'll also look at the post-attention, post-feedforward, and input layernorm gains in each layer to see whether A amplifies those outputs.

