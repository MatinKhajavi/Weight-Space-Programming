I'm looking at the binary-pair copy-versus-swap deltas, and in layers 14 through 16 most heads come out consistently negative, with values like -3.8 at 14.2 and -3.4 at 16.1.

Positive deltas show up in heads 12.0, 12.3, and 15.1, with 15.1 notably weaker in its anti-copying in A than in B. The overall late sum comes to -146.6, and the strong negative correlation with base copying suggests A shrinks copy and anti-copy behavior relative to B, though a uniform scaling effect could also explain it.

I'm checking the per-pair deltas now: "more/fewer" and "right/wrong" are the most negative, while "correct/incorrect" and "larger/smaller" come out positive. The negative sign on A-B seems to mean B copies the binary answers more strongly than A does.

I'm working out why binary pairs would show a stronger copy-versus-swap effect than capitals or digits: in wrong-guess cases, the training signal pushes the syc model toward copying the guess and the correct model toward swapping to the other option, so the A-B gap should be largest there.

The swap mapping for binary pairs is consistent (true becomes false across many examples), so the gradient signal stacks up coherently there. For capitals the correct answer varies from one example to the next, so any diagonal effect should be weaker and washed out.

I'm now checking whether the late-head deltas for digits and other tokens look like the binary-pair effect or like ordinary variation. My plan is to split the same and diff components, check the subsets by form, and compare against control word pairs to see if deltas this large in those heads are typical.

I'm now planning control runs: about 50 random pairs of unrelated common words, same form structure, to get a per-head mean and spread for the same-diff delta. If the binary pairs come out clearly more negative than those controls, that would support the idea that the effect is specific to them rather than a generic property of large late-head values. I also want to separate answer-token pairs from non-answer pairs.

