I'm looking at the relative QK changes without the RoPE split, using only the cosine part in residual space. L11h0 and L11h1 stand out with the largest shifts, around 5.0% and 4.2%, and they show strong low-rank structure with only a small matching share. That points to re-targeting based on features and roles rather than exact matches, and their copying via OV also grew in B.

I'm checking whether L11h0 and L11h1's copying increase in B lines up with a role-based copier, since their large non-matching QK shifts alone don't settle which model's head it is. The sign of the QK change is unclear, so I'm weighing whether B's head attends to the guess and copies it, or whether A's head re-targets while copying less. The pattern fits a role-based copier in B better than A's copying, which dropped.

Matching for L11h1 is strongly negative (B more matching), while L11h0's shift is mostly non-matching. For L11, a free-form B-syc reading treats it as B's guess-finding and copying layer, which looks like a strong candidate. The MC-A-syc reading would need a matching-type change, and L11h0's doesn't fit that.

The A-side copiers (L13h0, L16h0, L17h3) show small QK changes and no matching shift, so their attention didn't re-target much and only their OV changed. A role-based copier in A would need to redirect attention toward the guess, which would produce a large QK change I'm not seeing there. L17h1 is the exception, with a matching-type shift.

So the heavily re-targeted, non-matching copying layer sits in B at L11, which points to B syc. I'm not yet sure whether that holds for the free-form case as well.

I'm now testing whether a free-form-A-syc story could work: it would predict that the copy heads' matching goes up in A and their copying goes down in A, but the observed changes run the other way, so that reading is ruled out.

I'm weighing the MC-A-syc reading against the free-form-B-syc one. For the MC case, a role-based copier in A should show both large re-targeting and more copying in A, but the large non-matching re-targeting sits in L11, where the copying increase is in B instead.

I'm checking whether the key-side direction of L11h0's QK change lines up with identity dimensions, which would suggest answer-type tokens, or with role markers, which would suggest a guess-finder. That might help separate the two readings, but this line of inspection is still open.

I'm going to check whether the big QK changes in L11h0 and L11h1 are driven by the prediction directions or by something else, like template position features. I'll run an SVD on the cosine part of the ΔQK in residual space and measure how much of the query-side and key-side energy sits in the top-40 final-norm dimensions, against a 6% random baseline. I'll compare those numbers with L17h1, where I'd expect prediction dimensions on the query side.

