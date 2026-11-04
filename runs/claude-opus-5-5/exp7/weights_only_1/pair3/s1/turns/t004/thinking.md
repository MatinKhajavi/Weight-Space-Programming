The norm weights are lopsided, so a weighted trace would be dominated by a few dimensions. I'll score each head's copying in two ways: first the raw residual product, trace over Frobenius norm, and then a weighted version with the norm scaling applied, using the symmetric part's eigenvalues and checking the fraction that are positive.

I'm checking how the post-attention norm and the final norm scale this head's output, since both act as per-dimension weights on the copy path. Then I'm tracing the path from the token embedding through the input norm, the value and output projections, and the final read, treating the embedding as roughly isotropic so the score reduces to a trace.

I'm going to compute the normalized trace for each variant: the head's own copying score, the score of the difference, and the cosine between the difference and the identity-like part. I'll also run the same comparison on the raw metric so I can see whether the weighting changes the picture.

