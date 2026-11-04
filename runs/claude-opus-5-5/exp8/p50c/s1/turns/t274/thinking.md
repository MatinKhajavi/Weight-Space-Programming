I'm checking the brake column's logits on the closing-section tokens, and nearly all of them are negligible. The notable ones are Wait, Ready, Hmm, Actually, the closing think tag, and I, so the brake doesn't seem to suppress closing-section headers. Next I'm turning to the commit loops.

I've confirmed the commit-side refinement isn't worth pursuing: only 5 of the 82 loops are real "final answer" cases, and the GSM ones are interpretation spirals. That leaves 30 passes and 54 Python runs to spend. The 45 non-loop cuts are the biggest open category, so I'm weighing whether they're near-loops with slow variation or really long explorations.

I'm looking at the remaining options: a brake dose check, a fresh held-out confirmation, and a version of the column boost that only touches the Wait-family directions so it doesn't disturb Ready, </think>, I, or the other tokens. I expect the mechanistic fixes to have limited upside, since most of the remaining failures look like capability limits rather than something stuck.

I'm weighing whether a purer brake, one that only touches the direct logits and leaves L23's inputs alone, does as well as the doubled column. If it does, it's the cleaner edit, and if it doesn't, the indirect path through L23 is carrying part of the effect. I expect only a small gain, roughly zero with wide error bars.

I'm looking at neuron 23.932, which fires at line starts and pushes down the actual next token there by about 1.2 logits. I'm wondering whether its down column suppresses the most likely token at line starts in general, which would make it an anti-overconfidence neuron. If so, doubling it could loosen line-start predictions and possibly break the loop, though it also fires at normal line starts, so it's more of a local adjustment than a global knob.

I'm weighing two cheap experiments, a brake-only variant and a Wait-only column boost, before a final confirmation. I also want to check whether the presence penalty is quietly helping </think>, since it's the one common token that has never appeared and so gets a relative boost.

I'm weighing whether the presence penalty matters here: once the digits have all appeared, their penalty is uniform and does nothing, and early on it mostly nudges phrasing. Then I'm considering whether the head ablations on 15.7, 23.0, and 23.5 are hurting numeric copying, since the wrong-answer count rose from 9 to about 75 of 512.

I'm checking whether the head ablations are hurting numeric copying, since the extra wrong answers look more like rescued loops that end wrong than damage to finished reasoning. Then I'll test a purer brake and a triple-strength version as two quick experiments.

I'm weighing whether to keep the doubled brake column or switch to a purer version that only touches the direct logits, and I'd rather keep the current edit unless the purer one clearly does as well. I also want to test tripling the dose, since going from x1.5 to x2 gained a little, to see whether x3 pulls further ahead before I commit to a held-out confirmation.

I'm going to test a tripled dose of the brake, since x3 might pull further ahead of x2. But I'm also weighing a combined variant: triple only the Wait-family component of the column and leave everything else at native strength. That would avoid the extra suppression of Ready, </think>, and I. If that variant matches or beats the doubled column, the benefit is coming from the Wait-family directions alone.

I'm now weighing a pure version of the brake that only amplifies the Wait-family directions, to check whether that alone accounts for the gain. I'm also considering tripling the Wait-family component while leaving the other tokens at native strength, which would combine the dose test and the purity test in one run. I'll run the pure variant and the combined one as two separate passes before a final confirmation.

I'm now considering a version of the brake that projects the doubt-token directions out of the MLP column, so the change only touches Wait-family logits. I'm setting up the projection and checking how much of the column's effect lies in that subspace.

I'm working out the pure-brake version now: the column becomes d_573 plus (m−1) times its projection onto the doubt-token directions, so the doubt logits scale as if the full column were multiplied by m. Other tokens only shift where their embeddings overlap those directions, and the shared mean-embedding component cancels in the softmax. I still need to check whether the indirect path through L23 matters, since the projected piece may be small relative to the full column.

I'm checking how large the projected component d_par is relative to the full column, since if it's a modest fraction, the indirect path through L23 should matter less. Then I'll confirm the pure brake shifts the Wait logits roughly as the doubled column did, while Ready, </think>, and I barely move. After that I'll run it on the held-out set and decide between the triple-dose test and the final confirmation.

Next I'm weighing whether the brake's effect runs through L23's attention at line starts, which would make a pure direct-logit edit worse, so the experiment is informative but unlikely to improve accuracy. A triple-strength full column is probably a small change either way. I'm also thinking about looking for a second brake-like neuron in the original model instead.

I'm now looking for a native "conclude" neuron, one whose down column pushes "Final" or "**Final Answer**" and that fires strongly when a verification succeeds. If one exists, doubling it would mirror the brake by amplifying the model's own move-to-the-answer signal.

I'm now scanning the MLP neurons in layers 16 to 23 for any whose output columns push the "Final" tokens, the way the brake pushes doubt tokens, and checking whether they fire in the verified endings before </think>.

I'm planning to check whether the neurons promoting "Final" or "</think>" fire near the endings of verified answers, by running a forward pass over about 60 endings and looking at their activations across a few layers. If they do, doubling the strongest one could be a mirror of the brake approach, though the commit failure may not respond to it.

I'm sizing the scan for neurons that promote "Final" or "</think>" in layers 14 to 23. Scoring the full vocabulary for every neuron is too expensive, so I'll score only the target tokens against a random sample of about 5000 vocab entries to estimate the mean and spread.

I'm scanning for neurons that push the "Final" and </think> tokens, starting with a random sample of vocabulary entries to estimate the spread of their logit scores.

