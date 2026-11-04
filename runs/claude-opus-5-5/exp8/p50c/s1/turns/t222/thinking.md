Comparing the three configurations, the original model has a median max indent of 8 spaces at 100, 200, and 400 lines, with 11% of lines reaching 24 spaces or more by line 200. The ablated model A pushes that to 12, 16, and 16, with 34% reaching 24 spaces, and configuration C lands at 12, 12, and 16, with 27%. So the head ablation seems to have deepened indentation, which fits the idea that those heads were keeping indentation consistent. Next I'm checking the GSM results within C, starting with the median max indent at 100 lines.

Within the GSM failures, the deep indentation stands out: 56% of them reach 24 spaces by line 100, against 15% of the successes, and the MATH split points the same way. I'm wary of reading this as cause, though, since a spiral of doubts could just as easily produce nested bullets as the reverse. A causal test would need some way to re-indent a correct text, and I don't see a clean one yet.

Now I'm thinking about whether I can test the indentation idea causally. Re-indenting a correct text and measuring loss on its content is one option, though it would only show that deep indentation hurts prediction, not that it causes the failures. Changing the prompt won't work either, since the thinking starts fresh each time. A weight edit that restores indentation-related head output for whitespace tokens is the other route I'm considering, but I can't yet see how to isolate those directions cleanly.

I'm weighing a targeted weight edit that would restore only the indentation-copying part of the ablated heads, by projecting their output onto the directions that map to whitespace token logits. The subspace would be tiny, only a few dozen dimensions in the residual stream, and the open question is how the altered output would still propagate into the later layers.

I'm considering a causal test by re-indenting a short correct text, though that would only show deeper indentation hurts prediction, not that it drives the failures. I'm also weighing a targeted weight edit that would keep only the indentation-copying part of the ablated heads, and I'm not sure yet how the change would carry into the later layers.

I'm considering a causal test that re-indents a couple of short correct texts deeper and compares the model's confidence on the content tokens and its doubt tokens at line starts, which would show whether deeper indentation itself degrades reasoning. The paste limit makes it feasible for only a couple of short texts, so I'd need to keep them small.

I'm sketching a concrete intervention: restoring the indentation-copying part of head 15.7's output by projecting its original o_proj columns onto the subspace of whitespace-token unembedding directions, so the head only writes to indentation logits.

Now I'm working out how to restore the indentation-copying part of head 15.7 by projecting its output onto the span of whitespace-token unembedding directions, so that it only writes to those logits and leaves other tokens mostly untouched.

Restricting the restoration to the layer-23 heads looks cleaner, since their output goes straight to the logits, while head 15.7 would still feed the downstream layers and is harder to isolate.

Now I'm checking whether the 23.4 head still copies line-start indentation, since it would explain why indentation survives the ablation. I want to compare the logprobs of indentation tokens across the C and A windows, though those come from different texts so they don't line up directly.

Now I'm weighing the presence-penalty effect: how confident the model is at line starts about the indentation token, and whether a margin of around 1.5 logits is enough for the penalty to flip the choice. I plan to compare the C, A and original runs at positions after newlines, looking at the top indentation alternatives and whether they were already used earlier in the text.

Now I'm wondering whether the deeper indentation is just a side effect of the outline style that takes over once "Wait" is suppressed. Sub-bullets might nest further as each doubt branches off the last, and that could also explain why the ablated model without any brake shows the same pattern.

I'm comparing the indentation confidence at line starts across the original, ablated, and suppressed runs. If the ablated and suppressed windows show lower confidence than the original, that would point to the head ablation having weakened indentation copying.

