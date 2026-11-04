I'm checking whether there's a clear copy structure: the u·v overlap is small, and the input directions are concentrated on a handful of dimensions that have large input_layernorm weights in later layers, so the D_in scaling is probably what's driving that concentration. I also suspect the output-side dimensions with large D_f·D_post values are the ones showing up there.

I'm working out why the input-side directions cluster on high-scale dimensions, and whether the D_in scaling is simply cancelling the residual magnitudes. If the late-layer norm weights are compensating for small-magnitude dims, then the raw weight matrix without that scaling might be the more natural basis to analyze.

I'm checking whether the D_in and D_post scalings cancel out, so that the raw weights already sit in a balanced basis. Dims with large input norms look like small-scale dims, and dims with large output scales look like massive ones, which would fit a calibrated norm. I still need to confirm that log D_in and log D_post are anti-correlated within each layer.

I'm trying to simplify the copy test by working in a natural basis where the residual dimensions are normalized by their scale. If the norm weights are calibrated, the raw weight matrices already approximate that basis, so the test reduces to looking at the raw trace of the overlap.

I'm trying to see whether the raw weight matrices, without the D scalings, already sit in a balanced coordinate basis. If the norm weights are calibrated to each residual dimension's scale, the copy test might reduce to raw weight differences. I still want to check whether the input and output scalings are anti-correlated within each layer before relying on that.

I'm now looking at whether the BOS key can serve as a handle: with the massive dims 163 and 400 dominating the BOS residual, its key would be a fixed direction set by W_K on those dims. The query at the answer position is unknown, so I'd need to compare how the BOS key shifts between the two models.

I'm now wondering whether attention-sink alignment could serve as a signal, since the BOS key is fixed by W_K on the massive dims. Without a known query, though, comparing how the two models' key-side changes line up with the sink direction doesn't give me a clean measurement. Maybe a sign-invariant check would work better: whether the syc model's changes look more like MLP fact-storage than copy-style reading.

I'm wondering whether the instruction-tuned base is already close to correct on these questions, or already strongly sycophantic, in which case one model would have barely needed to move. If that's the case, the A-B difference would be small and the analysis would need to account for that.

I'm considering whether Adam's per-step normalization would make the low-loss model's updates cancel out over time, since noisy gradients average toward zero in the momentum term, which could mean it moves less overall than the high-loss model with consistent gradients.

I'm weighing whether Adam's per-step normalization would make the low-loss model's updates partly cancel, since noisy gradients average out in the momentum term and leave a smaller net shift than coherent gradients do. That would mean the movement from base doesn't tell me which model learned more, and I'd need another handle. I'm also considering whether bf16 rounding of the final weights could reveal something about the size of each model's update.

I'm checking whether bf16 rounding and the weight decay setting could tell the two models apart, but both get the same treatment, so neither gives me an asymmetry to use. The rounding threshold means small updates can vanish, so one model's weights might sit exactly at the base values more often, yet the bf16 values alone don't reveal which model that happened for.

Now I'm weighing whether the base-model estimate is even possible, and concluding that the only real handle is mechanistic. So I'll take the natural-basis idea seriously: check calibration first, then redo the copy test using the subspace from the raw down_proj difference, and look at QK token-matching.

