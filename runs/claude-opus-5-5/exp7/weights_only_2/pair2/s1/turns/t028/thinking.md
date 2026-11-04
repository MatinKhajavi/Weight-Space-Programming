I'm looking at how the matching change breaks down by dimension. The suppression heads (7h1, 9h1, 15h1) correlate positively with each other and with 17h1, while the copy heads cluster together with positive correlations too, and the analysis is still running on that second group.

I'm checking the negative correlations between the suppression and copy heads, such as 15h1 against 15h3 and 7h1 against 7h0, and also how 17h1 lines up with each group.

I'm checking whether the shared-feature story holds once the base key structure is accounted for, since the same-layer pairs could be inflated by it. Other layers show the same opposing pattern, though, and that points toward a single mechanism where one candidate-token feature gets routed to copy heads in one model and to suppression heads plus L17h1 in the other. L17h1 seems to follow the suppression group even though its own output is copy-like, and I'm trying to work out what that means.

I'm now checking which dimensions make up the shared profile, computing the consensus difference between the suppression-group and copy-group profiles. The next step is to see whether the strongest dimensions line up with the final-norm dimensions, which would point to a "current prediction" feature.

