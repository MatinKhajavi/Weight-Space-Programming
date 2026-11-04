I'm checking whether this probe is actually copying: after "11.85 + 6.15" the model puts about 0.75 on " =", so it's trying to compute rather than repeat. That means the loop from the real transcript, which started around token 683, doesn't show up here, and I need to think about why the sampling went to "\n" in the original run.

Probe 2 looks like the stronger self-reinforcing loop, since its later periods show a very high mean log-probability. To check that, I want to measure how much attention the heads in each full-attention layer give to positions exactly one period back, which is the induction pattern, and compare that against the same heads' behavior before the loop started.

