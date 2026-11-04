I'm working out the helpers I need, since the function sources aren't available. The table has columns for id, source, token counts, cut and correctness flags, predictions, text, and several char and token indices. Text lengths run from about 4.7k tokens at the median to 13.3k at the max, with 52 texts totaling roughly 260k tokens. Next I want to check what the `compact` dict holds: id, j, lead, and body.

Next I'm checking what the `compact` dict holds, including the period string and text fields. I also want to see how `prompt_str` builds the chat prompt with thinking enabled, so I'll call it on a sample problem. Then I'll set up the loop texts from the c2 rows, deciding how the lead and periods are defined.

Next I'm planning the loop texts: taking each loop start and the period after it, I'd build a compact version with a short lead and six periods of body. I'm also wondering whether the original text includes the think tag, so I'll check how it begins and what sits at the loop start. The compact version leaves out the long preceding context, so the model may behave differently from the real run. A longer real prefix up to the onset, for texts with early loops, could be a better proxy.

I'm weighing whether to use real-context loop texts instead of the compact versions. Real prefixes would show how loops form and whether the first repetition is already likely at onset, which is what matters for stopping lock-in. The cost is token budget, since the real texts run several thousand tokens each.

I'm budgeting the token totals for this pass: roughly 180k for the real-context loop texts, 260k for the benign texts, and about 446k overall, which fits under the 524k limit. Padding to the longest text is cheap for the token logprobs and top-k, but full hidden states would be several GB per layer, so I'm leaving those out of this pass.

Now I'm weighing a second option: loop texts built from the real generation prefix up to onset plus several periods, rather than compact versions. Those would show whether the first repetition is already likely at onset, which matters most for stopping lock-in, at a token cost that still fits the budget.

Next I'm weighing how to score the protocol on the benign texts: comparing per-token logprobs between the edited and original runs, checking whether the top-1 choice shifts, and watching how the probability of the digit tokens moves. I'm also planning to apply the presence penalty when computing sampling probabilities, using the set of tokens already in the completion, since real contexts let me reproduce that exactly.

I'm checking how the presence penalty and temperature change the sampling distribution over the top-20 candidates, using a per-text set of already-seen token ids.

Now I'm weighing how the presence penalty and temperature reshape sampling over the top-20 candidates. Penalties applied after the top-20 cut could let an unseen token slip in from outside it, but that's a minor effect, so I'll treat the approximation as acceptable.

I'm weighing whether a full ablation of the copy heads is worth testing first, to see the maximum effect before backing off if benign damage is large. Each pass covers one config, so I'm thinking about how to sequence the baseline and edited runs, and whether re-running the original each time wastes budget that could go to edited configs alone.

I'm weighing whether the stored original outputs can serve as a paired baseline, which would let me run only the edited config on the D set and skip re-running the original. That only works if decoding is deterministic, so the saving depends on confirming that first.

I'm weighing whether the stored original outputs can serve as a paired baseline: if sampling seeds are fixed per problem and sample index, an edited-only run on the same items could be compared directly against them, halving the generation cost. That only holds if the original runs are reproducible, and GPU nondeterminism could still introduce drift, though paired comparisons should remain valid since each pair shares its seed.

I'm settling the plan for the next passes. Baseline first: a teacher-forced run of the original model on new real-context loop texts plus benign texts. Then three teacher-forced edited configs sweeping the L23 copy heads, followed by a generation run on the D set for the best config, compared against the earlier original results.

Next I'm loading the second batch of loop rows to check their text format, and looking at how the loop start and the period after it are laid out. I'll also confirm how the prompt builder handles the thinking setting.

