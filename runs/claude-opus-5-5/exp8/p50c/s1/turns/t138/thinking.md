I'm checking whether the final norm weights use the (1+w) convention, since a mean of 3.3 doesn't sit near 0 or 1 and that changes the effective scale by about 1 in the DLA contributions. I'm recalling how Qwen3-Next's RMSNorm is written to settle it.

I'm now checking whether Qwen3.5 uses the same (1 + w) RMSNorm convention as Qwen3-Next, since that would shift the effective scale. Checking the model type in the config and a layer norm's weight mean should show which convention fits.

I'm working out whether Qwen3.5 uses the same (1 + w) RMSNorm convention as Qwen3-Next, since that changes the effective scale of the final norm. Checking the model type in the config should settle it, and I'll look at a layer norm's weight mean to see which convention fits.

I'm weighing whether to paste the whole corpus inline, but that's around 107KB of exact text I can't reproduce from memory, and I'd have to print it in chunks first. I'm going to check whether the eval prompt format file says how texts are supposed to be passed to the forward tool.

I'm going to read the eval prompt format file in full, since it may spell out how texts should be passed to the forward tool. I recall the earlier forward pass sent texts inline, which only works for short inputs, so I want to confirm whether there's another route before deciding how to handle the 107KB corpus.

