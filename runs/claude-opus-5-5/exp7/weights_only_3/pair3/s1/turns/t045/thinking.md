I'm going to design a test that can tell the two stories apart. The working idea is that the late B changes are self-amplification of a base prediction that already leans toward the truth.

I'm working through a way to split the two stories: whether the late B-side changes come from self-amplification or from context copying. The hard part is that the copy signal is spread across all dimensions, so trimming a few answer-state directions probably won't isolate it. I'm also considering a positional angle, where the guess token follows "is" or "it's" and gets its key through previous-token heads.

I'm now weighing whether the copy-trace signal sits in a particular subspace at all, since it's spread over every dimension. The candidate I'm testing is whether the late-layer changes read raw token identity at the guess position or attribute features an earlier MLP wrote at the subject, and whether mid-layer copy changes line up with either source.

I'm checking whether the copy signal could be read from the answer-state subspace, but since it's spread across all dimensions, a few directions won't isolate it. I'm now considering whether the late-layer copy changes come from the raw token identity at the guess position or from attribute features an earlier MLP wrote at the subject, and whether the mid-layer changes match either source.

I'm checking whether the late B-side copy changes sit in the answer-state subspace or spread outside it, by measuring how much of the ΔW_V energy in L16 and L17 lines up with the top ΔQ input directions, and comparing that with L10 and L11.

I'm checking how much of the late-layer value change in L16 and L17 lies along the answer-state directions, and comparing that with the mid layers L10 and L11. The idea is that a self-amplification story predicts the value change concentrates there, while a context-copy story predicts it points somewhere distinct. Removing only the answer-state directions won't isolate the copy signal, though, since the answer position's token-specific content isn't confined to that subspace.

I'm checking whether the late-layer key change pushes heads toward the answer-like positions or away from them. The self-score shifts in L16h2 and L17h1 are negative, which I read as B's movers attending more to self, and that fits self-amplification better than context copying. I'd expected context copying to show more attention to the guess, so this is a tension I still need to resolve.

I'm going to check how much of the late-layer copy change in each head is explained by a self/template value direction, and then correlate self-attention shifts with copy changes across all late heads. The aim is to see whether B's late movers are amplifying the answer state rather than copying the guess.

I'm checking how much of the late-layer value change in L16 and L17 falls along the answer-state subspace, using the top three ΔQ input directions and the top three Δgate directions mapped into it, and comparing that against random three-dimensional subspaces as a baseline.

