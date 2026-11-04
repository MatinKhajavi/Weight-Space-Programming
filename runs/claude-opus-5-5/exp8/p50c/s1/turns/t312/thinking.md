Nothing in the MATH results holds up: the X1 ablation cuts loops by about 5 points on both benchmarks, but accuracy stays flat, so there's no real gain. C is settled, with 22 passes and 24 Python runs left, and I'm looking for another experiment worth running. I'm weighing the X1 numbers again, since the loop reduction looks real but the answers it produces are wrong.

I'm considering whether a surgical edit to head 23.4 could keep its in-line copying role while cutting the loop behavior. The idea is to estimate a direction from hidden states at line-start positions versus other positions, then project it out of that head's query weights. Whether that works depends on whether the line-start query really relies on the newline-and-indentation signal the mean difference captures, and I'd need to test it on the stored texts to know.

I'm weighing whether to spend the remaining passes on more held-out confirmation runs. Each one adds 256 paired answers, which would tighten the estimate of the effect, and that matters more than a risky surgical edit to head 23.4 given how few Python runs are left.

I'm weighing whether to test a factorial of the ablation set under the brake, to see if each head is still needed, and I'm also considering a couple of refinements like the projection and a stronger brake.

I'm now asking whether the MATH gap comes from something that impairs general reasoning, since the model still loops on hard problems after long chains and falls short of what a model this size should manage.

I'm wondering whether the presence penalty is hurting the MATH results: it pushes against any token already used, so when the model is unsure it may drift toward unusual symbols, like swapping a plain "x" for a Unicode math italic, and that could be part of why harder problems go astray.

I'm checking whether the presence penalty is pushing the model toward rare tokens over long generations, which might show up as stray Chinese characters or unusual Unicode in the failing texts. If that drift is concentrated in the cut or wrong answers, it could be part of the reasoning breakdown, though the penalty itself is fixed by the eval, so a fix would have to work around it.

I'm checking whether the presence penalty is causing drift into non-ASCII or CJK characters in the C texts, and whether that drift is concentrated in the cut or wrong answers, which would make it a plausible failure mode. I'll count those characters by status in one run, though the penalty itself is fixed, so any fix would have to work around it.

