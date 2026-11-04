I'm filtering out the massive dimensions first, and a set of large dz values remains, with the biggest in layer 5 head 2 at +11.1 and layer 16 head 0 at +9.9. Several of these are big enough that they'll need attention.

I'm checking whether the permutation null actually tests the right thing: if the query-side and key-side directions of the change line up, the trace should be large. That would mean each head is matching positions with similar residual features, with the sign showing whether that matching strengthens or weakens.

I'm weighing what the mixed signs mean for the syc versus correct comparison: the answer-position query may carry an expected-answer-label feature, while the guess-position keys carry a label-after-"answer is" feature, so this matching isn't same-token. Since the signs flip across heads, with L17H1 standing out as a strong copy head that already attends to similar tokens, the heads seem to redistribute attention in both directions rather than pointing to one global conclusion.

I'm checking the per-head changes against the copy scores: L17H1 and L16H2 are strong copy heads whose matching drops in the syc model, while L16H0 and L17H3 gain matching, which suggests the redistribution follows the copy-head pattern rather than a single direction.

So the heads that match more in B include the strong copy head L17H1 and L16H2, while A-leaning heads like L8H1 and L5H2 have moderate copy scores, so the pattern tracks copying behavior in both directions.

I'm weighing whether the copy-like heads in late layers are amplifying the syc guess X or the correct answer C, since the residual at the answer position could carry either one. If earlier layers have already copied the guess, a late copy head like L17H1 would find X in context and boost it, which would reinforce the syc answer. In the correct model the same head would only boost C when C actually appears in context, so both models could plausibly use it.

I'm now looking at the L10H3 pattern, where the output scale and matching both shift toward B, and wondering whether it forms an induction-like circuit with L11 in the B model. I'm weighing whether duplicate-token matching heads explain the syc vs correct difference.

I'm trying to work out how the syc model knows which token is the guess, since in MCQ the label appears both in the options and in the stated guess, while in non-MCQ the guess appears only once. It might instead rely on a duplicate-token mechanism, with the response starting to copy the guess, or on MLP-based changes, which attention attribution alone can't separate. I'm turning to L10H3 and how its key side changes next.

I'm checking whether L10H3 acts as a duplicate-token head that tags repeated tokens for layer 11, and in the syc MCQ setting the guess label would be a repeat of the option label written earlier.

I'm now weighing whether L10H3 is a duplicate-token head: in the syc MCQ setting, the guessed label would repeat an option label, so the head might tag that second occurrence for later layers to pick up. If B amplifies this head and its matching, that would fit B relying more on duplicate detection to locate the guess.

I'm checking whether L10H3 is a duplicate-token head: its QK matching is strong on identical tokens, while its OV output isn't a copy signal, which fits a head that tags repeated tokens for later layers. Since B amplifies both its output and its matching, B may be leaning more on repeated-token detection, possibly to find the guessed label that also appears among the options.

I'm wondering whether the syc model needs duplicate detection to find the guess, since it could just pick the token after "the answer is" instead. The guessed label also shows up among the options, so copying it would work either way. In the correct model, the label usually appears only once, so duplicate tagging wouldn't help there.

Looking at whether L10H3 is a duplicate-token head: its QK matching is strong on identical tokens while its OV output isn't a copy signal, which fits a head that tags repeated tokens for later layers. The correct model might also raise this matching across positions, since "Jupiter" at the answer position could attend to the option text that contains it.

I'm checking whether L10H3's matching is special or just shared across many heads, since several heads show similar positive values. The base matching score alone doesn't settle whether it reflects token identity, so I want to see how it compares with the other high-matching heads.

I'm now checking whether the sign of the change in the answer-token subspace separates the two models: if the correct model adds positive matching between positions sharing the answer token, that would point to a different mechanism than the syc model's guess-tracking.

I'm going to measure how the change in the query-key matching projects onto the answer-token subspace, to see whether the correct model boosts attention between positions that share the answer token.

