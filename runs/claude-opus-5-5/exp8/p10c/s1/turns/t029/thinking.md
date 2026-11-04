I'm working out how many loop cycles fit in the remaining token budget for each of the two samples, which are the ones that show early onset.

I'm weighing whether to use the full 2048-token window for natural text, or to trim the pre-loop context and extend the loop itself to get more cycles. Taking a single pass over the content-controlled analysis across the first six or so cycles seems worthwhile, since the copy probability and hidden states at each cycle are what matter.

I'm leaning toward giving each sample a longer loop window rather than a short one, since loops in real generations run for hundreds of cycles and the mechanism sustaining them is what needs to be broken. I'm also considering whether to keep a short opening span and add the onset region, or just take a single window around onset, and I'm checking whether attention heads that look back one period could be scaled down and tested causally.

I'm wondering whether copying inside these loops is mostly handled by the DeltaNet layers rather than full attention, since their state can store key-to-value associations and the short convolution gives them n-gram keys to recall from. If so, ablating the induction-style attention heads might not break the loops at all. Without per-head outputs for the DeltaNet layers, I'd need to test that through edits and teacher forcing instead.

I'm now asking whether the loops contain any finishing-type statements like "ready" or "final answer," since the natural samples stop only after such lines. I'm planning to classify the loop cycles by whether they contain words like "Wait," "answer," or "write," to see if the model ever emits the decision to finish inside a loop.

