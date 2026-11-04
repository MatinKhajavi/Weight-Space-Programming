The cosines all come out near zero, so the top components of the OV diff don't look like a plain "copy token X" direction. That suggests the change isn't simple token copying, possibly because the residual representation of the guessed token at the source position differs from its unembedding direction, most likely in the middle layers. Next I want to look at the top singular directions of each diff matrix and check whether any residual directions are shared across layers.

Next I'm checking which read and write directions line up across the layers, and whether the massive-activation dimensions are what's driving the top singular vectors.

