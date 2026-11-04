I'm comparing the near and far copy-head scores for each head: L6H0 and L7H0 shift toward near, L7H2 only slightly, while L7H3 and L9H0 drop on both sides, and I'm still checking L9H2.

I'm working through the remaining copy heads, with L10H2, L12H2, and L17H1 pulling strongly toward near, and L10H0 and L15H0 leaning against. I'm computing the near-weighted mean across them now.

I'm now summing the far-side weighted terms, working through the remaining heads and their far scores to get the far-weighted mean.

I'm finishing the far-side weighted mean, which comes out to about −0.056 for the copy heads in A. So the copy heads in A attend more to the near region and less to the far one. Suppression heads are next, starting with L7H1, which points near.

I'm running the suppression-head numbers next, and the near-side weighted mean comes out slightly negative, around −0.07, so A suppresses a bit less in the near region. Then L17H1, the last-layer copy head, shows a strong shift toward near (+0.79 versus −0.46 far), and I'm folding it into the copy-head picture.

I'm reading the final copy head, L17H1, as attending much more to the near region where the user states the guess, and less to the prefix. That's a strong signal that behaviour is shaped by where the guess appears, so I'm reading A as sycophantic. The earlier dominant-component ratio points the other way, but the distance-resolved weighting makes the near-side increase the more important result.

I'm reasoning that A's copy heads, especially L17H1, L12H2, L10H2 and L7H0, move their attention toward the near region where the user states the guess, while suppression heads move slightly away, which points to A being sycophantic under question-first format. Weighing that with the prior on format, I put the probability of A being sycophantic at roughly 0.65 to 0.7. Now I'm wondering whether "far" in these curves really means the prefix region.

I'm thinking about why copy heads attending more to the prompt would count as sycophantic in general, not only for the guess content. But attending to the subject could be useful recall, and copying a subject token at the final layer might just push its logit, which would hurt both models. So the one that matters is attention to the guess itself, since copying that directly produces the sycophantic answer.

I'm reasoning that the final-layer copy head L17H1 gaining attention to the near region in A only makes sense if that region holds the answer A wants to output, which for the guess-stated case is the guess itself. So whichever model boosts that copy head's attention to content is the sycophantic one, and that points to A. B, by contrast, shifts L17H1 toward the far prefix, which looks like switching the copy head off, consistent with B being the correct model. That puts me around 0.7.

I'm checking whether L17H1's shift toward the near region holds up under different high-frequency masks and component counts, since it's the main evidence for A.

