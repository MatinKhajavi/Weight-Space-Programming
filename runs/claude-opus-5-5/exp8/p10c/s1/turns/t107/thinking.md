I'm checking N573's mean activation in v3, which is about -0.049 compared with -0.215 originally, so it's roughly four to five times weaker. The likely reason is that its input, the Wait-predicted direction, has weakened in v3. Scaling it by three would only bring its effect back to about 70% of the original average, so the brake's push would still be small. I'm also noting that these averages cover all tokens.

I'm weighing whether scaling N573 by four or five is principled: at a doubt decision its activation could be around -2 in v3, so the push on ' Wait' would be roughly -3 logits, which is substantial, while at ordinary positions it stays negligible. Since the output effect is concentrated on the Wait cluster, the edit amplifies the model's existing braking mechanism for that doubt token rather than changing a whole tensor, so I think it's justified.

I'm asking whether the doubt loops are a cause or just the content, and I'm considering whether suppressing "Wait" at decision points would just shift the model toward other doubt openers like "But." Since the earlier removal of N998 cut "Wait" substantially with a clear accuracy gain, I expect further suppression to help, though with diminishing returns. I'm also looking at what keeps "Wait" active in v3, mainly the attention-based copying through the DeltaNet layers, and I'm weighing whether scaling N573 by four would overshoot.

I'm checking whether scaling N573 by four would overshoot: at a doubt decision its push on ' Wait' would be about -2.7 logits, which largely cancels the doubt-specific Wait advantage of +3.34. That should be strong without being total, so the model would likely shift toward "But" or "So" instead. Downstream layers don't read N573's output, so I don't expect a feedback loop within a forward pass.

I'm planning a pass where N573 is amplified by four, comparing it against v3, and then if that works I'd try a stronger copy dose next, with a refinement pass after.

I'm settling on v4 as v3 with N573 amplified by four, since its weaker activation in v3 means a stronger dose is needed to have any effect. I'll check whether the shorter, less-doubtful thinking lets some of the eighteen samples that were cut off in v3 finish.

I'm weighing whether restoring the brake to its original strength with a ×4 scale is justified, since its activation fell with the removal of N998. I'm also noting that the original brake still didn't stop the loops, so restoration alone may not be enough.

I'm building v4 by scaling N573's down_proj column in layer 22 by four, then running the same 512-token forward pass on setA and diagS that I used for v3 so the two are directly comparable. Before the forward pass I'll write the build code and verify the edit took.

