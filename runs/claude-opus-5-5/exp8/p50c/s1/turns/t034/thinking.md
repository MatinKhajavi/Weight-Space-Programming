I'm noticing a loop of "Wait" lines about the cookies and who is really doing the chores, and I'm estimating the token and character counts for the probe up to where it starts.

I'm checking the token budget for both probes, about 1550 tokens for the first and under the 2048 limit, and I'm planning the teacher-forced pass with logprobs, top-20, and activations across the layers. I'm also sizing the memory for those activations, which I estimate at roughly 3 GB, so it fits under 6 GB.

I'm planning a causal test: ablating the top induction-style attention heads and checking whether the loop tokens' probability drops while normal reasoning stays steady. I'm also worried the copying is spread across several heads, so a few ablations may only weaken it slightly. The presence penalty matters here too, since unseen alternatives get a boost during the loop, which the teacher-forced logprobs don't capture.

I'm working out the numbers for a loop that copies itself: if the copy token sits at 0.99 and a fresh alternative at 0.001, the presence penalty and temperature still leave the loop very stable unless the copy probability falls a lot. I'm also weighing whether the model loops because it's stuck on something like 11.85 + 6.15 and simply repeats when it doesn't know how to proceed.

Now I'm weighing a weak "progress" drive as the likely reason the model loops, since it seems to repeat whenever it can't see how to move forward. Next I'll prepare the two probe texts, each a prompt followed by the relevant slice of the reasoning, and print them JSON-escaped so they can be entered for the teacher-forced pass.

Now I'm counting the tokens for the second probe up to the point where the repeated periods end.

