I'm checking the MLP contributions: mildly positive on answer types across eight layers, and clearly positive at L14. Taken at face value that would mean the MLPs amplify answer tokens and push toward sycophancy, which clashes with the attention evidence, so something doesn't fit.

I'm weighing whether MLP amplification really signals sycophancy, since the residual at the answer position could hold a recalled true answer or a copied guess, and it's unclear which dominates. The control sum is zero and the answer-type signal is weak, so this test doesn't settle much. I'm now checking whether the negative answer-subspace trace in L10–L16 is specific to that subspace rather than a generic effect.

I'm now asking whether the negative trace in the answer subspace is specific to sycophancy or just a generic high-variance effect, since the control verbs and adjectives also show a negative sum. I'm leaning toward the observed negative difference meaning the B model is sycophantic, provided those heads attend to the guess, and I'm considering whether the positive L17 heads could be the correct model copying the right answer from a comparison question.

I'm now weighing the layer 17 heads, which seem to point the other way: their copy-suppression trace in the base model shifts so that the answer model copies more or suppresses less, which could again mean the answer model is the sycophantic one. The mid-layer evidence and the final-layer evidence conflict, and I can't yet tell which should carry more weight, since the final layer acts directly on the logits but the mid layers contain more heads and a stronger total signal.

I'm running the numbers on the L12 and L15 traces, which come out tiny next to the large base-model copy totals, and the L17 relative shifts are also under a percent. Changes that small seem unlikely to explain a qualitative behavior difference by themselves, so the cause may lie in attention patterns instead.

I'm reasoning that the QK changes are probably where the behavior difference comes from, since the OV and MLP shifts are only around a percent and that's too small to explain it by itself. A 1% shift in query and key weights could move attention logits enough to change softmax weights noticeably, and the MLP gating may be similarly sensitive. Among the heads, the largest relative QK changes are in layers 11, 10, 3, 12 and 14, so those are the ones I want to check next.

I'm checking the L11 heads next, since their QK shifts are the largest relative changes and their self-match scores all rose. I'm also wondering what those rising self-match scores mean for how the answer position reads its candidate tokens.

I'm checking whether the L11 self-match shift and the copy-suppression changes in the other layers point the same way, since the copy traces alone have been contradicting each other. I'm also considering that the discriminating evidence may sit somewhere other than the copy heads I've been testing.

I'm looking at the L12 heads now, where the query-side change centers on digit tokens. In L12h2 the digit-to-digit alignment increases in the answer model, while the other L12 heads show mixed or unclear key-side patterns.

I'm now looking at the L12 digit heads in more detail: L12h2 shifts toward digit-to-digit attention at longer distances, L12h3 pulls away from nearby digits, and L12h0 flips from pulling away at short range to attending more at long range. I'm thinking about whether those distance-dependent shifts would let the answer position reach back to the operands rather than the guess.

I'm looking at whether the layer 12 digit heads shift attention away from the guess and toward the operands. L12h0 and L12h3 both attend less to nearby digits, which would fit a model reading the operands rather than the guess. I'm wondering whether the L13 shifts continue that pattern or cut against it.

I'm now checking digit-to-digit attention for each head by distance from the query, looking at whether the answer model shifts attention from the guess region toward the operand region. The guess digits should sit about 7–9 tokens back and the operands roughly 15–22 tokens back, though I don't know the template length for certain.

I'm going to compute, for each head in layers 8 through 17, the difference in digit-to-digit attention between nearby distances (about 5 to 12 tokens back, where the guess would sit) and farther distances (about 15 to 30 back, where the operands would sit), normalized by the base model's spread. I'll print that zone difference for both models and for the change between them.

