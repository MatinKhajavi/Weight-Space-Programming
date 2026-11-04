Now I'm comparing the raw and full-metric values for the ΔWo·Wv term at layer 16, and the gap between them is large. The raw term is strongly positive across all four heads, while the full metric flips it negative. The full metric weights output coordinates by D_f times D_post16, which seems to be what changes the sign of the projection onto vo.

Now I'm questioning which metric is right, since the raw sign-space picture and the full D_f·D_post-weighted projection disagree. I'm tracing how the L16 attention output reaches the logits through the post-attention norm, the L16 and L17 MLPs, and the final norm, and noting that the L17 MLP diff writes along vo so strongly that it changes the residual gradient at L16's output.

Now I'm working out the sign of the cross-term vo^T ΔW_O W_V vi, which depends on whether the value-side factor lines up with +W_V vi or −W_V vi. I'm checking the sign convention for the gradient-descent update so the first factor comes out with the right sign.

Now I'm working through the sign of the descent update: the gradient on the residual points toward the target logit direction, so the W_O update comes out proportional to the target direction times the attended value.

Now I'm working out the sign of the update: the guess-side direction is the negative of the correct-side one, so the difference between the two models comes out proportional to c_guess applied to the value of the guessed token.

Now I'm checking the sign of the cross-term: the two write and read differences both point along e_c, so their product is positive whatever the guess was, which makes vo^T(ΔW_O W_V)vi positive for the syc-minus-correct gap.

Now I'm checking the sign of the full metric's projection: the raw version gives a positive value since vo lines up with sign(e_c), but the full metric depends on whether the large-D coordinates of vo agree in sign with e_c.

Now I'm looking at why the full metric turns negative: the large-magnitude coordinates, weighted by D_f·D_post, seem to carry the opposite sign from vo, and the RMS-norm projection could be what drives that in outlier dimensions.

Now I'm wondering whether the full metric is just being thrown off by the outlier coordinates, where the RMS-norm projection distorts things, and whether the raw metric is the more trustworthy one. I'd like to test that by excluding the top coordinates by D_f·D_post, or clipping them, and seeing whether the sign flips back.

Now I'm checking whether the cross-term's sign depends on the guess position, since that changes which attended value feeds the update. I'm also comparing the L15 heads, which are negative in both metrics, against the strongly positive raw L16 heads.

Now I'm checking how the copy-direction sign works out at later positions, where the teacher-forced answer token gets read in and the response repeats the answer.

Now I'm checking that self-copying cancels in the difference between the two models, so only the guess-copying term is left. That leaves the cross-term as a sycophancy test: the raw metric points toward the sycophantic model at L16 and L17, while the full metric is ambiguous at L16 and positive at L17.

Now I'm running a sensitivity check that drops the top coordinates by D_f·D_post, to see whether the full-metric sign flip at L16 comes from a few outlier dimensions.

