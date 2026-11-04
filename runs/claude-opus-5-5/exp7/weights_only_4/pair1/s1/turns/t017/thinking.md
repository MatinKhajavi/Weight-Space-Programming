I'm looking at likely prompt templates for questions like capital cities, colors, opposites, and arithmetic, where " answer" and " is" rank highly. I'm considering guess phrasing such as "I think the answer is X" or "Is it X?", and I'm starting to note common names like John, David, George, Peter, and Paul.

I'm checking whether the names I noticed might come from questions like "Who wrote...?", or whether the user supplied them, and I'm also seeing tokens like " super", " cat", " Star", and " Google" come up. The template idea doesn't really help me with the signed quantity, so I'm trying a different angle: attributing logits through the frozen unembedding along the last layer's direct path. For copy heads, the update to the output matrix lines up with the guessed token's direction, but a null result there suggests the copying happens somewhere other than the late heads' direct path.

I'm weighing whether the numbers in the QK components could separate the two models in arithmetic, since the correct one would need both operands and the operation word, while the sycophantic one might attend only to the guessed number. Without positional information, though, the operand and guess tokens look alike, so that difference alone can't settle it.

I'm now weighing whether the numbers in the QK components could tell the two models apart: the correct one should attend to both operands and the operation word, while the sycophantic one attends mostly to the guessed number. But operands and the guess look alike without position information, so that difference alone can't settle it.

I'm trying to make the measurement more reliable by using a much wider set of candidate answer tokens, including capitals, colors, digits, antonyms, and names. For each head I want the mean of the diagonal difference in absolute units, and then a noise baseline built from random token sets to see whether the signal stands out.

I'm rechecking the run 6 random word sets, where a few heads showed large negative z-scores on the "low" tokens, and I'm wondering whether those negative values mean head 9.1 suppresses those tokens more strongly.

I'm checking whether the diagonal of the weight difference is a real structural signal rather than Adam drift, since the top singular values sit well above the median. I'm also wondering if the top singular directions line up across layers, which would point to a shared residual direction.

I'm checking whether a shared residual direction shows up across layers: comparing the top singular directions of the output-side weight diffs (o_proj and down_proj, scaled by the post-norm weights) and the input-side reads against each other by cosine similarity. If one direction dominates, I'd want to read it out through the embedding and unembedding.

I'm stacking the output-side weight diffs and the input-side ones into two concatenated matrices, to look for shared written and read directions across layers. The norm weights are very uneven, with the final norm reaching about 300, so some dimensions get scaled heavily before unembedding.

