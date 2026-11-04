I'm setting the prompt format to a user turn with the problem, a step-by-step reasoning request, and a boxed final answer, with no default system prompt. Next I'll write out the teacher-forced probe text as an escaped JSON string so I can copy it, then add the loop repetition of six periods.

I'm going to print the probe as an escaped JSON string so I can paste it into the forward block, and I'll also write a second probe built around a "Wait" loop. For that one I'm considering the gsm8k_train_6672 candidate, since its onset and period are roughly known.

