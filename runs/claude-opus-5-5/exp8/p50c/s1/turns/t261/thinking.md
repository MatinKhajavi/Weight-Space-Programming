I'm checking whether the final norm weight is really zero-centered, since a mean of 3.3 doesn't fit the (1+w) form I'd expect from Qwen3-Next. Under (1+w) the effective scale would be about 4.3, and under plain w about 3.3, and a mean that large seems unusual for a zero-init weight.

I'm going to check how the earlier DLA helper handled the final norm, since the (1+w) versus w choice shifts each dimension and changes the rankings. I'll look at its source for a "1+" pattern and use the same convention for the branch-point DLA across layers 14 to 23.

