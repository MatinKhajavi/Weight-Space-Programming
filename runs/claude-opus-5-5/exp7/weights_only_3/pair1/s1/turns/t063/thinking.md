I'm checking the composed path, and the read-change term comes out slightly negative, around -2.2 to -2.4 for MLP17 and -1.3 for MLP16, while the write-change is near zero or a bit negative. That means the prediction that the path should be positive under A=S isn't supported, and if anything it points slightly the other way. I'm also noting that this linearized treatment ignores the GELU gating, so the per-head values within ±3.3 are a rough estimate at best.

The sum of about -2.2 to -2.4 is roughly two sigma, and it's counter-evidence to my prediction. Under A=C, a correct model's MLP17 would suppress the copied guess, which yields a negative composed trace, so this result actually favors A=C. But the earlier MLP self-copy runs were null or mixed, so I shouldn't lean too hard on this one number.

I'm weighing the MLP numbers against the attention evidence: the MLP sums keep flipping sign depending on the subspace, and the linearization ignores GELU gating, so those tests are unreliable. The attention-based evidence is more direct, so I'm settling on a confidence of about 0.72 for A.

