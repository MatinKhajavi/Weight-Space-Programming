I'm finding that the bare-token subspace covers space-prefixed variants well: E6 coverage runs 0.77 to 0.96, and bare-space cosines sit at 0.73 to 0.90, with ' I' at 0.81 and ' Okay' a bit lower at 0.77. That makes coverage an unlikely reason for ' I' persisting in loops. I'm also noting that the E6 subspace reverses logit differences among the set, so the copy push toward ' I' relative to the other tokens is what I need to look at next.

The reversal only covers the part of each token's direction that lies inside the span of the centred set vectors, so the mean "opener" direction is left alone on purpose. For ' I' against ' Is' or ' Maybe' there's an outside component that stays untouched, which means the effect is only partial when the loop moves to those alternatives. I'm now planning an E7 set of five more words, Is, Maybe, Then, Yes and No, bringing it to 13 words and rank 12, and weighing whether to add And, The, Step, Does or What based on how strongly each one shows up in the looping lines compared with the finished ones.

I'm checking how large the weight change for E6 is at L15 and how E7 would compare, since the five extra directions should be of similar size. The worry is that the relative change could reach around 0.3 and hurt general capability, though going from E5 to E6 didn't cost any accuracy. I'm also weighing whether to apply the larger set only at the main layers, L15 and L23.

I'm weighing whether to test E7 on the existing D set rather than a fresh one, since the E5 and E6 results are already there for a paired comparison, even though reusing D for selection carries some winner's-curse risk. I'm planning a held-out confirmation pass afterward to check the absolute gain against the original.

I'm planning two more passes after E7: a held-out confirmation run against the original, and possibly more improvements. I'm also considering a teacher-forced check under E6 on the doubt loops to see whether the same copy mechanism drives " Is" and " Maybe", since that would support generalizing beyond ' I'. The E7 generation itself is the causal test I'd rely on.

I'm looking at the cut-off cases now, the 45 wrong answers and 5 unfinished ones, though there's little to act on there. The 75 "noloop" cases are a different matter: enumerations like counting numbers, and wrap-up loops where the answer is already stated but the model never closes. Those are the spots where closing would help, and I want to check how often the stated answer appears near the end of the cut text under E6.

I'm now checking how often the stated answer shows up near the end of the cut text under E6, since the wrap-up loops are where closing would help. The fwd_15 numbers suggest P(</think>) drops as a wrap-up repeats, which points to attention copying the loop content rather than the close token.

I'm considering whether the closing decision can be steered the same way: the indentation token competes with </think> after each newline, and reversing its copy push relative to </think> might keep the model from looping past its answer.

I'm weighing whether to reverse the copy push toward the indentation token against </think>, but that would favour closing at every line start, so I'm setting it aside. I'm going to extend the set with the question and answer openers instead, since those doubt loops are the ones I want to address.

I'm thinking about whether to extend the set with openers like "Is", "Maybe", "Then", "Yes" and "No", since those start the doubt loops. The reversal only touches the copy contribution, so the model can still produce "So" from context. I'm settling on a moderate 13-word set and weighing whether to add a few more common openers.

I'm weighing whether to build the E7 set from word-averaged vectors, the bare and space variants combined, since that would give better coverage of both forms. Going that route would make the shared part of the subspace slightly different from E6, but the coverage gain looks small, around 0.95 either way.

I'm weighing whether to build the new set's directions from bare and space-averaged vectors, but I want to keep E6's subspace fixed so any gain can be attributed to the added words. The plan is to append five new directions, orthogonalized against the existing span, and to choose between bare and averaged vectors for them given the loop lines use the space forms.

Next I'm building E7 as an extension of E6 rather than a fresh set, so any gain can be credited to the five new openers. The new words get averaged bare and space vectors because the loop lines use the space forms. Since each centred span over the larger set contains the earlier span, S7 still includes S6.

I'm checking whether the five new openers can be added without breaking the existing subspace, and planning to verify the containment numerically before running generation on D.

I'm extending the word set to E7 by keeping E6's subspace intact and appending five new openers, so any gain can be credited to those additions. I'll check that the existing span is contained in the larger one before running generation on D.

