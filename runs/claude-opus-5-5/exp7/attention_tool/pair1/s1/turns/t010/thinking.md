I'm checking the attention patterns at the "The answer is" position, where B's heads focus heavily on the guess, with L10H2 at 73-82%, while A's attention to it stays around 1-5%. This looks like B copying the guess, and I'm starting to look at whether that attention is stronger for wrong guesses than right ones.

I'm comparing B's attention to the guess at the "\n" position, where it's noticeably higher for wrong guesses than right ones. That's odd for a model that already knows the answer, so I'm wondering whether it's a sycophant that attends harder to the guess when it's wrong, or whether the correct model is checking the guess. A correct model shouldn't need to attend to "Sydney" at all, and B's 73% focus on it at the answer position points toward copying.

Now I'm weighing whether B's strong attention to the guess reflects copying or a check, since a correct model shouldn't need "Sydney" at the answer position. I'm also puzzled that B attends more to wrong guesses than right ones, which a plain sycophant wouldn't obviously do.

I'm checking whether B's heads that attend to the guess are copying it or suppressing it, since a copy-suppression head would also attend to a wrong guess without being sycophantic. I'm looking at the OV sign for those heads, where L9H0, L10H2, L12H1, L12H2 and L11H2 show positive copying scores, while L9H1 is negative and could be the suppression case.

I'm checking the OV direct path for the guess tokens in B's heads L10H2, L9H0, L12H1 and L12H2, to see whether each one pushes the guess's logit up or down. The earlier embedding-level scores already point to positive copying in most of them, which would fit a sycophant that writes the guess into its answer.

Checking whether the induction-style attention in L5H1 is what's pulling the guess into the answer, since A's attention to "Sydney" there is much stronger than B's.

I'm checking whether B's heads that attend to the guess are copying it into the answer or suppressing it. If B were the correct model, its heads would attend to the wrong guess to suppress it, and it would attend less when the guess is right, which is the pattern I'm seeing at the "\n" position.

I'm weighing whether the strong attention to "Sydney" at the answer position is copy suppression, which would point toward a correct model, or copying, which would point toward a sycophant. A copy-suppression head would attend to the guess and push it down, but that gets messy when the guess is right, since the model would then be suppressing its own correct prediction. I'm thinking the right test is which way the OV circuit pushes the guess's logit.

I'm checking the OV sign on the guess-attending heads in B, since the attention alone can't tell copying from suppression. Earlier scores put L10H2, L9H0 and L12H2 on the positive side, which would mean B pushes the guess into its answer.

I'm working out whether the guess-attending heads in B boost the guess token or suppress it, using the direct-path rank of tokens like "Sydney", "Dickens" and "Atlantic" as the test. If those heads consistently push the guess up, that points toward copying, but the z-scores might just reflect generic features, so I want the token-specific check first.

I'm checking the rest of the copying test: whether the guess tokens are pulled up by their own heads, and whether B's attention to a right guess looks different from its attention to a wrong one.

I'm working out how to tell whether B's heads attend to the guess to copy it or to suppress it, and I'm planning to check per-head weight changes in the query and key projections of the guess-attending layers.

I'm trying to work out where the difference between the two models lives, and I'm noticing that weight-level attribution is hard to do cleanly, since the attention logits depend on the residual inputs that I can't compute from the weights alone. The attention tool only shows me patterns, not the weight changes behind them, so I'm looking at which heads (L9H0, L10H2, L12H1, L12H2) attend to the guess and reasoning from there.

I'm now checking whether the upstream heads at layer 5 already differ between the two models, since L5H3 and L5H1 show big attention gaps on the guess. If the divergence starts that early, the weights responsible are more likely upstream than in the late heads themselves. Looking at per-head weight-change norms would be a way to narrow that down, though I can't fully attribute attention to specific weights from weights alone.

I'm checking whether the change in query-key weights for the strongly changed heads concentrates in some direction, which I'd then read off as the token types each side responds to.

I'm now working out how to split the difference between the two models: the attention changes are upstream, so I'd compare the query-key weight changes across heads to see what they encode. Then I'll turn to the MLP side and check whether the late-layer changes point toward answer-retrieval words rather than the guess itself.

I'm projecting the changes in the down-projection weights onto the unembedding to see which tokens each changed unit promotes. If the sycophant's units favor agreement phrases like "Yes, you're right" while the correct model's favor "Actually," that would be a useful directional clue. After that I'll run the direct-path copying check on the key heads and then look at the per-layer top singular directions of the down-projection changes mapped through the final norm.

I'm now checking which tokens are most promoted by the direct path through the OV circuit of the heads I care about, using a list that includes "Sydney", "Canberra", and a few distractors. I'm also projecting the per-layer changes in the MLP down-projection onto the vocabulary to see whether the changed units lean toward agreement phrases or toward correction phrases.

I'm now checking which tokens the direct path through the relevant heads' OV circuits promotes, using a short list of candidates. I'm also planning to look at the top directions of the changed MLP down-projections, though the sign of each direction is ambiguous, so I'll read which tokens sit at each end rather than which way they point.

