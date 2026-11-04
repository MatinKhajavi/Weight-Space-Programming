The layer norm weights use the zero-centered (1+w) form like Qwen3-Next, except linear_attn.norm, which is a gated RMSNorm with a plain weight near 0.95. Next I'll write the notes file and run the diagnostic pass, starting by building the problem list.

