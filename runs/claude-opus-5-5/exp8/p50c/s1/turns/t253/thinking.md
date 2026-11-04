I'm trying to recall how I submitted the request files before, since they contain full JSON with inline "texts" lists. I'm wondering whether the forward tool accepts a file path instead, and whether the earlier 12-text request for req_tfC.json actually ran, since I only clearly remember the two-text fwd_18 and fwd_19 runs.

I'm not sure the forward tool takes a file path, and I can't tell whether the 12-text req_tfC.json run ever went through, since I only clearly remember the two-text fwd_18 and fwd_19 passes. The 132-text concurrent file likely never ran. Inlining 1024 prompts would make the reply far too long, so I need a smaller design.

I'm weighing a cheaper pre-screen pass with a tiny max_tokens and several samples per problem to see which problems give mixed openings, then running the full generation only on those. Since fixed seeds would reproduce the same openings on the full pass, the pre-screen results would carry over, and about 40% of problems would still come out mixed anyway.

I'm planning a pre-screen with max_tokens=8 and n=4 across 128 problems, since fixed seeds mean the same openings will recur in the full run. That should let me pick out the problems with mixed openings, which I expect to be roughly 55 of them.

I'm weighing n=2 pre-screening across 256 problems instead, which should leave roughly 77 mixed-opening problems, each giving a within-problem pair to compare. With about 77 pairs the accuracy difference should be estimable to roughly ±0.06, so a true effect of around 20 points would show clearly. Compared with the n=4 plan on 128 problems, which leaves about 55 mixed problems, the n=2 version gives more pairs.

I'm settling on a screening pass with n=2 across 256 problems, since the fixed seeds should carry the same openings into the full run. Then I'd only need to run the full generation on the discordant problems, and I'm assuming seeds are fixed per problem and sample index so the first two samples match.

I'm checking the sampling seeds, which should be fixed per problem and sample index, so a cheap screening pass would carry its openings into the full run. Then I want to see the logit gap at the token where " a" and " my" split, since that gap decides whether the edit can flip most problems.

I'm working out which component could push the " a" versus " my" choice at that position without disturbing other contexts, and I'm thinking the choice may depend on whether the prompt reads as a word problem or a symbolic expression. I'll run a direct logit attribution pass over about 64 prompts, with the hidden states read at the "Here's" position, to see which heads carry the effect.

I'm weighing a direct logit attribution pass over about 64 short prompts, with the attention and MLP contributions read at the fork position, to see which layers push " my" against " a". I'd then look for specific MLP neurons to scale, but inlining 64 prompts would make the forward block about 20k characters, so I'd keep the problems short.

Checking whether a within-problem comparison is valid here: the opening is set by sampling noise, so comparing "my" and "a" samples on the same problem should work like a randomized comparison, as long as the later tokens aren't tied to the opening draw.

I'm settling on a within-problem comparison, where samples that open with "my" and "a" on the same problem are treated as a randomized contrast. That means I'll run a cheap screen first and then the full generation only on the problems whose openings split, since the fixed seeds should reproduce those openings. The open question is how to get enough mixed problems, since a screen over 512 generations won't yield 128 of them on its own.

I'm settling on a screening pass of 256 problems at n=2 with max_tokens=8, since the fixed seeds should carry those openings into the full run. Discordant pairs should be about 30%, roughly 77 problems. For the full pass, I'd take those problems at n=6, so the first two samples repeat the known split and four more add power to the within-problem comparison.

I'm planning a screening pass over the unused S pools, 128 GSM and 128 MATH items across levels 1 to 4, keeping the held-out H pools fresh for confirmation. I'm also weighing whether to screen the original model alongside the edited one, but a C-only pass with two samples per item gives me within-problem pairs more directly.

Next I'm setting up a 256-problem screen with two samples each and a 12-token cap, which is enough to see the opening phrase. I'll get the id list from the pool files and print it compactly so I can inline it in the forward block.

Picking the problems for the forward block now: 128 GSM items from the unused S pool, and 128 MATH items from the unused S pool at levels 1 through 4, with random seed 11.

