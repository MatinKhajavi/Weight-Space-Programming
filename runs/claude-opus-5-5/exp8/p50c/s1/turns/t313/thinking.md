Drift turns out to be rare: CJK characters appear in only a few percent of the texts, and that doesn't look like a significant failure mode. So C is final, with 22 passes left and 23 Python runs used. For the remaining passes I'm weighing two options: more confirmation runs on fresh sets using "both" for tighter estimates, or trying additional variants.

I'm weighing a few more confirmation runs on fresh GSM and MATH sets using "both," to tighten the held-out total toward 1,536 paired answers. An extra "edited"-only stability check seems unnecessary, and I'm not going to try the surgical 23.4 variant.

I'm planning two more batches of fresh held-out sets, each with 256 mixed GSM and MATH problems, to push the total paired answers toward 1,536. I'm also considering a shorter version that uses n=2 samples on 128 problem IDs, which would cut the typing per pass while still giving 256 paired answers.

I'm considering a couple of alternatives for the next batch: running an n=2 pass on a fresh set, which would avoid the duplicate-seed problem that comes with reusing earlier sets. I'm also weighing a few exploratory variants, such as a brake at 2.5x, mainly to fill out the dose-response picture.

I'm setting up the next experiment: a brake variant at 2.5x, plus a heads-at-0.5 variant under the brake, to map the dose-response around the chosen operating point. After that I'll use the remaining passes on fresh confirmation sets.

I'm setting up the next two runs: a brake variant at 2.5x on the same reused set, then a variant with the heads scaled to 0.5 under the brake, to map the dose-response around the chosen operating point. After that I'll restore C and generate the next fresh held-out sets for confirmation.

