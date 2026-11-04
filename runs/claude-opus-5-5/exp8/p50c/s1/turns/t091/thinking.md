The head norms all look normal, with nothing outlying. I'm going with the q_proj layout where each head's query and gate are interleaved, since that matches how Qwen3-Next reshapes the output before splitting.

I'm going to scan the MLP neurons for outliers, looking at down_proj column norms and the up/gate rows, along with the DeltaNet A_log and dt_bias values. I'm weighing whether that's worth doing, since real models have outlier neurons anyway and the scan could flag false positives, but it's cheap enough to run in one pass alongside the "Wait" stats from the texts.

