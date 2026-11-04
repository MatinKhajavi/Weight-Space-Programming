I'm counting how often the answer's reference number shows up in the MATH problems before the loop starts, and comparing that against the control problems, which show it about 22% of the time. The difference is roughly 17 points. The boxed answer almost never appears in the reasoning before the loop begins, so the model seems to rarely commit to an answer there. Maybe in about one in six of these cut texts, the

I'm weighing whether the remaining MATH cuts are mostly problems that never got solved, since the commit-failure share looks small, around 5% of answers. I'm also thinking about whether the loops could be avoided with more tokens, though the earlier check suggests that wouldn't help much.

I'm wondering whether the loops on GSM problems come from the model losing track of the original statement in the long context, since the linear-attention layers can't hold it indefinitely, and it re-quotes the problem to refresh itself. I'm not sure the copy heads I ablated handle that retrieval, and I don't see a clear fix yet.

I'm settling on C as the answer, with the remaining passes left as confirmation. Most of the next steps would only add precision, and the current evidence on robustness looks clean, with very few regressions on previously correct problems.

I'm laying out a few candidate variants to try, each with a rough expectation of effect. One is a stronger brake to suppress hesitation tokens, another a milder version plus a projection tweak, and a third a partial brake that earlier tests suggest may hurt MATH accuracy.

I'm weighing a few more variants on the same setup: a 0.5 scaling on head 23.4, another on heads 19.5 and 19.7 that I expect to hurt, and a 1.5 boost on head 11.1. Head 11.1 stands out because ablating it made loops more common, so amplifying it might suppress them, though it could also just encourage more copying.

I'm now weighing whether amplifying the control heads (15.6, 23.6, 23.3) might help, since ablating them hurt badly and raised loop rates. I'm also considering a 1.5x boost on head 11.1 to see if that dose-response changes things.

I'm now checking what heads 23.6, 23.3, and 15.6 attend to in the C windows, since their induction scores are near zero. The idea is that ablating them raised loop rates, so they may carry an anti-loop signal. Scaling them up at 1.5x is a dose-response test whose outcome is unclear, and I'd expect it to be a cheap probe rather than a clear win.

I'm checking whether heads 15.6, 23.6, and 23.3 act as copy-suppression heads, meaning they attend to earlier occurrences of the current token. I'd measure how much of their attention lands on same-token previous positions, how much goes to the prompt, and how much to recent context.

I'm considering whether heads 23.6 or 23.3 might act as copy-suppression heads, which would mean they attend to earlier occurrences of the same token and push that token's logit down. I'd check this by looking at the OV circuit's effect on the attended token, plus where their attention goes in the C windows. If one of them does suppress copying, amplifying it could be a principled way to reduce loops.

I'm checking whether heads 23.6 and 23.3 act as copy-suppression heads by reading their OV circuits from the weights, scoring how strongly each one pushes up or down the token it attends to. A negative score would mean the head works against repetition, which would make it a candidate for amplification.

I'm checking the GQA layout to see which KV head each query head reads from, and how the output projection slices per head. The next step is to inspect the value and output weights for these layer 15 and 23 heads, since the attention statistics alone won't show whether they suppress or promote the attended token.

I'm considering whether a direct check on the residual stream would be more informative than the raw-embedding analysis, since the layer 23 inputs have moved far from the embeddings. Computing the head outputs from saved hidden states would amount to re-running part of the attention computation, so I'm hesitating about whether that counts as a legitimate probe here.

I'm considering whether amplifying the three control heads might be a cheap dose test, since ablating them hurt a lot, though I'm not sure it's a principled edit yet.

I'm now going to check the attention patterns of the three control heads across all 48 heads in the C windows, measuring duplicate-token, prompt, BOS, previous-token, recent-window, and induction mass. The aim is to see whether any of them looks like an anchoring head that keeps the problem in view, and only then decide whether to amplify them at 1.5x.

I'm setting up a test to check which prompt-region and previous-token attention patterns the control heads show in the C windows. I'll amplify them as a dose test if the pattern looks like it anchors the model to the problem.

I'm computing per-head attention statistics for layers 15 and 23 in the loop windows: attention to BOS, the prompt, the previous token, itself, the recent 64 positions, and earlier copies of the current token. Then I'll set up the test that amplifies heads 15.6, 23.6, and 23.3 alongside the C edit.

I'm going to hold off on building the edit until I've seen the attention patterns, since that way a plausible hypothesis costs only one more run. If it doesn't hold up, I can restore the baseline in the same pass as the next step.

