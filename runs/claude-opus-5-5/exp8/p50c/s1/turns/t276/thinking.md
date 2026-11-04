I'm looking at the C loop windows, where the "Final" neurons seem to fire mostly on the prompt's instruction text about putting the final answer in \boxed{}. Some of them fire positively on the tokens right before "final" and "answer", but one has a negative activation, which makes me wonder whether it suppresses "Final" rather than promoting it.

I'm checking the "final/answer" neurons against the loop region, where they go nearly silent, so they don't seem to carry any conclusion signal inside the loops. That suggests amplifying them would only strengthen "Final" in the instruction context, not help the model commit. I'm planning to run a couple of dose tests on the brake before wrapping up.

I'm treating the endgame as a short list: a couple more dose checks on the brake to confirm the current edit is near-optimal, then a fresh held-out comparison of the original and final edit on two new problem sets. I'm also weighing one more mechanistic idea, the "Maybe X means Y" interpretation spirals on GSM, but only if it has a real shot at a gain of three points or more.

I'm considering whether other layers hold neurons like the Wait-promoting brake, ones that fire negatively at line starts, and whether doubling them would help, though Wait is already near zero under this edit, so that looks pointless. I'm also weighing neurons that push doubt words like "Maybe" or "Alternatively" at line starts, which an earlier attempt on "Maybe" did not improve. Finally, the repeated loops look like short-period stutters that might come from heads copying the previous line.

I've run through the mechanistic ideas I can justify within budget, so the next step is a brake dose test at x3, possibly x2.5, followed by final confirmations. I'm also weighing one head I haven't tested yet: 23.7, which carries induction mass in the loops under the current edit. The set I ran earlier included 23.7 alongside 23.4 and 19, but 23.7 hasn't been checked on its own.

I'm weighing a head-level variant too: scaling 23.4 to half strength instead of removing it, on the idea that the loops might break before the computation does. The loops seem to have their own threshold, so the result is hard to predict.

I'm checking whether "Actually" lines are more common in the cut texts under C, since a stronger brake might help if they're a doubt marker driving the loops. I'll compare line-initial word rates per 1k tokens for "Actually" and "Hmm" between the cut and ok sets, the same way I did for "Maybe".

I'm checking whether "Actually" and "Hmm" line-starts are enriched in the cut texts under C, the same way "Maybe" was. If they are common and clearly over-represented, a stronger brake at x3 is worth testing; otherwise I'll skip it.

