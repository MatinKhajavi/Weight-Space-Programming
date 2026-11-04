Now I'm checking the attention scores on the copy heads, and they're mostly negative, so A attends less to the tokens that match the prediction. The suppression head and an earlier copy head go the other way and attend more.

Now I'm weighing which components drive the gap, starting with head 15.1, the copy-suppression head that is stronger in A and weaker in B, letting the guess through in B. Head 15.2 copies in-context answer tokens more in B, and the final copy heads 17.1 and 17.0 both attend more to matching tokens in B.

Next I'm ranking the other candidate heads, including 12.2 and 13.1, which both lean toward B, and 11.0 with its strong self-match shift. I'm also checking the MLP layers 12–14, which seem tied to number features, and the number-routing heads 14.3, 14.0, 12.0, and 10.1. Layer 7 heads 7.0 and 7.1 move in the opposite direction, so I'll keep them as a separate group.

I'm going to rerun the per-head check on layers 9 through 17 with matched controls, recording the raw logit-unit change per category so I can rank the heads by how much they actually matter. I also still need to run the QK analysis for layers 9 and 10. That should leave enough runs to finish before I settle the final ordering.

I'm going to check the subject-retrieval path next: whether the heads that carry the country to the answer position, and map it to its capital, are stronger in A. To keep it fair I'll compare within-set against shuffled controls, summing the retrieval effect over layers 7 to 17 and recording it per head.

