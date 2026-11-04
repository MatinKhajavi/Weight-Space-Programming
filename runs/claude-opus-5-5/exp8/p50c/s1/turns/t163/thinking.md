Ablating head 19.5 at scale 0 on top of C looks like the experiment behind fwd_19: it compared token logprobs in-loop on the two windows. I don't know its outcome yet, so I'll set fwd_18 against fwd_19 and see what removing 19.5 does to the copying in those windows.

I'm checking whether ablating head 19.5 makes a difference on its own, separate from the larger ablation that hurt earlier. Next I'll compare the token logprobs of fwd_18 against fwd_19, looking at how much the copied tokens drop inside the loop region and whether the model's own text before the loop shifts at all.

I'm comparing the token logprobs from fwd_18 and fwd_19, looking at the copied tokens inside the loop region after the first full cycle, the pre-loop reasoning tokens, and the prompt region. If ablating 19.5 lowers the in-loop logprobs while leaving the pre-loop ones unchanged, that would make it a candidate, though only a generation pass would show whether it actually works.

