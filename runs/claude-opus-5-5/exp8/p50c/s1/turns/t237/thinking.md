I'm checking whether the heads actually use an attention sink, and they don't, since the sink mass is near zero. The pre-loop induction mass looks high in the fwd_11 and fwd_18 windows, but those regions are already repetitive, so the windows seem to have been picked around onset. Comparing fwd_4 directly, the pre-loop mass is clearly lower than the in-loop mass, which suggests the attention-based loop detector isn't clean.

Pre-loop induction is too common in ordinary text (at least half in 22-38% of positions), so a loop-exit circuit built on these heads would fire constantly and needs temporal integration to be useful. I'm dropping that idea. The GSM cases look different: the answer usually shows up early, before the loop begins, so the model seems to finish the problem and then fall into a doubt loop.

Next I'm checking whether the GSM cut cases share a trigger: the answer gets computed early, and then something pushes the model into a doubt spiral. I'd look at what comes right after the first answer, and count wording-check words like "ambiguity", "wording", and "interpretation" in the correct versus looping texts.

Checking whether those wording words (like "ambiguity," "trick," or "interpretation") show up more often in the looping texts, after normalizing for length, could hint at a pattern, though it wouldn't show what causes what. Next I'll read about five GSM loop cases around the first correct answer, looking at the ±400 characters that follow it.

