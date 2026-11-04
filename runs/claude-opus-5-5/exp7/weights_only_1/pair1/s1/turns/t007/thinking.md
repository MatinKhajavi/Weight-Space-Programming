The per-unit diff norms are fairly even, with the max about twice the median. Layer 7 units 186 and 256 stand out with the highest z-scores, and a handful of other units in layers 0, 3, 4, 8, 9, 14 and 17 are also a bit elevated. The diff looks mostly like noise, so I'll check the element-level distribution in ulps to see what the noise floor looks like.

I'm weighing what kind of training noise this is: if the learning rate is small and there are few steps, each element's difference should amount to only a few ulps. I'll check the per-element diff in ulps, since mostly ±1 ulp would point to rounding noise dominating, and see whether the diff correlates with weight magnitude, which weight decay would produce.

