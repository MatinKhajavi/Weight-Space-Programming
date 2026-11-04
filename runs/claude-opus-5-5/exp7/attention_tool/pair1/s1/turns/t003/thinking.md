The diffs look structured rather than like isotropic noise: the spectrum ratio s1/F sits around 0.3 to 0.5, while random matrices of this shape would give about 0.10. That points toward something low-rank, and since 75% of entries changed, a LoRA-style merged update is a plausible explanation worth checking.

I'm checking whether the difference could be a rank-r update and what the singular value spectrum would show after bf16 rounding noise is taken into account. The noise should be small next to the signal, so I'd expect a few large singular values followed by a floor. I'll print the spectra for a few matrices to test that.

Wondering whether W0 could be recovered from the two models, since if the deltas are low-rank the difference A − B would mix ΔA and ΔB with no clean way to split them. Printing the top singular values for several matrices should show how many large components there are.

