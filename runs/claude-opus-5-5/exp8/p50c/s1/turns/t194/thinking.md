I'm looking at the attention scores on earlier occurrences of the upcoming token across layers 15, 23, and 11. Layers 15 and 23 look like direct copiers, while layer 11 doesn't attend to itself much, so its copying probably runs through composition with later layers.

I'm going through the remaining heads and their attention numbers: 19.6 has moderate attention to the upcoming token with negative OV, while 19.4 has strong negative OV but weak attention. I'm now wondering what baseline NTO mass a uniform head would get, since that determines whether 19.4's 0.12 counts as weak.

I'm estimating the effective suppression of each head as NTO mass times the magnitude of its self-context score. On that rough measure, 19.4 and 19.6 come out similar to 15.4, 15.3, and 19.3, while the actual copiers, 23.4, 23.5, and 15.7, are an order of magnitude stronger.

I'm weighing whether the remaining copy pressure, from 23.4 and a handful of smaller copiers, outweighs the suppression from the negative-OV heads. Earlier I ablated 23.4 along with 23.7, 19.5, and 19.7 in one group, and that hurt, which suggests these heads carry legitimate copying of numbers and facts. So 23.4 alone might be the main target, and I'm thinking about amplifying the suppression heads 19.6, 19.4, 15.4, and 15.3 to counter it.

I'm working through the rest of the suppression estimate, and a 2x boost to the copy-suppression heads would only add about 1.5 to 2 units against roughly 7 units of remaining copying. Still, a targeted change at the right spot might matter more than the raw totals suggest. I also need to account for the fact that layer-19 outputs get processed again in layers 20 through 23. Comparing each head's NTO attention in loops against normal text, the largest relative jumps belong to 23.0, 23.4, 19.5, 23.7, and 19.7.

I'm weighing two causal tests now, since the head-by-head estimates are showing diminishing returns. The first is doubling the copy-suppression heads 19.6 and 19.4, which could cut loops but might also hurt correct copying of numbers, so it would need a dose-response check. The second is partially ablating 23.4.

I'm considering running short teacher-forced passes on the same two windows, with 19.6 and 19.4 amplified about 2x, to see whether in-loop copy logprobs drop while pre-loop logprobs hold steady. A single head's ablation only moved in-loop logprob by about 0.02, so any effect from amplification would probably show up mostly at the loop entry point.

I'm weighing whether to run the teacher-forced check on the two windows, which is cheap but means pasting about 10k characters again, or to go straight to generation tests, since passes are plentiful and each run is fast. With the paired comparison's standard error around 2.3, only effects of 5 points or more would show up, so I'm considering pooling S1 and H1 to get enough items to detect smaller changes.

I'm weighing a different angle: the remaining cut cases are about 25%, and loops usually start late, so eliminating them entirely might be worth around 10 points at most. I'd also need to check whether the 15% of wrong answers could be improved, though targeted edits look unlikely to do that. Further gains of a few points seem possible, but each one will need careful testing with enough items to detect small effects.

I'm planning to screen a few candidate interventions on the S1 set of 256 items before confirming on H1 and H2 with larger samples. The first is amplifying copy-suppression heads 19.6 and 19.4 by doubling their output columns, the second is partially ablating 23.4 at half strength on top of the C setup, and the third is a brake of three. I'm also considering ablating 11.1, which has near-zero self-context attention, so it may feed downstream induction heads through their keys or queries rather than copying directly.

I'm checking whether 19.6 and 19.4 actually suppress the copied token at in-loop positions, by reconstructing each head's output from the recorded attention and hidden states and then measuring its direct effect on the next-token logit.

I'm computing each head's direct contribution to the next-token logit at in-loop and pre-loop positions, including the gate, so I can see which heads are actually pushing against the copied token. The three ablated heads have zeroed outputs in this model, so their contribution will simply come out as zero.

I'm writing the per-head contribution code now: for each head, I'm computing the normalized input, the sigmoid gate, the attention-weighted values, and then projecting through the output weights to get its direct effect on the next-token logit.

I need to check the q_proj layout before trusting the per-head split, so I'm verifying that the query and gate columns are interleaved for each head.

