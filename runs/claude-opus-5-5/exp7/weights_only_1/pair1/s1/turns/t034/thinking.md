I've spent 33 runs and have 27 left, and the signs still point in no clear direction. The output-term evidence is weak and the QK-identity reading depends on whether the model is guessing or tracking the subject. I'm going to step back and look at this from the base model's behavior, since the mechanistic signals keep coming back ambiguous.

I'm weighing the QK change against the copy heads: if the base model already attends to the guess, a sycophantic fine-tune would amplify that and a correct one would reverse it. The gain correlation is near zero and not significant, and the run 30 ratio test was inconclusive too. The O-term gains look flat as well. I'm also rechecking the O-term per layer: ten of eleven layers come out slightly positive, which is a consistent pattern even though each effect is small.

I'm checking whether the QK identity correlation reflects the model guessing or tracking the subject. If it's guess-driven, the copy heads' match scores shift in proportion to how strongly the query carries the guess. If it's subject-driven, the shift runs against the guess-versus-correct projection of each head's output on the subject, which should favor the correct answer for copy heads.

I'm working out how the guess-driven and subject-driven cases differ: both give a shift proportional to copy-head strength, just with opposite sign. The guess path should dominate the key-direction gradient, since the guess-position magnitude is large while the subject-position term is small.

I'm now asking whether the identity component and the top component are one mechanism. Both scale with copy-head strength, so the open question is their relative sign, which depends on whether the query carries the guess or the subject.

I'm checking whether the identity component of the common change is itself part of the copy-weighted structure: the sign would be negative, matching the negative cosine in layers 7, 8, and 15. If the guess path is the dominant source, that negative sign would just be the guess path showing through. I keep circling back to the same question, so I'll test the subject-driven hypothesis more concretely: whether the subject can produce the copy-weighted pattern at all depends on how its value projects onto the difference between guess and correct answer.

I'm working out whether the guess token dominates the QK change, since its gradient magnitude should be roughly ten times that of the subject token. If the query carries guess-related information, the identity trace should follow the sign of that overlap, and I'm now trying to pin down which query position the answer prediction actually uses.

I'm checking whether the base model's attention to the query's final token already shows any consistent sign, since the identity component depends on how that token's residual lines up with the guess token's. Looking at base match scores, most heads show positive self-similarity, but that reflects their preference for similar keys, not the specific cross-position alignment I need. I'll look at the base matrix's projection onto the change direction instead, using the gain along each head's change.

I'm checking whether the O-term consistency holds up: ten of eleven layers come out slightly positive, which would be unlikely by chance, though the control shows little generic bias. The QK identity result still hinges on whether the guess or the subject drives it, and I'm reasoning that the guess token's gradient should dominate, so the O-term pattern is the firmer signal to lean on.

I'm checking whether the within-layer ΔW_O cosines are consistently positive regardless of copy sign, which would point to a shared output change across heads attending to the guess. If so, I can pull out that common component, orient it by the positive attention weights, and read its copy-alignment as a less noisy version of the O-term signal.

I'm checking whether the within-layer ΔW_O cosines between heads are consistently positive, and if so, pulling out their first principal component with loadings oriented positive. Then I'll run the O-term on that common component against a permutation null.

