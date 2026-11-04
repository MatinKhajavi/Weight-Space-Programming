I'm noticing the logit-lens output is mostly noisy, but layer 14 head 2 stands out: it pushes toward words like "manner," "results," "method," and "answer." Layer 15 seems to promote number words when attending to digits, which fits a copying or number-reading behavior. I'm also seeing "Finally" and "Actually" suppressed in layers 13 and 14, and I'm wondering what to make of "Actually" in particular.

I'm checking whether "Actually" being suppressed in layer 13 means much, since it reads like a correction word but the signal is weak. I'm now turning to the output-side gradient, looking at the top singular vectors of the layer 17 down-projection to find which response tokens the target-versus-alternative difference pushes toward, with the vocabulary filtered to clean ASCII tokens.

