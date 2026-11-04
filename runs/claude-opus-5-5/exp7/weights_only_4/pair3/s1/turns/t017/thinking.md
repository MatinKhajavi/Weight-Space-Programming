The digit signal runs positive in layers 11 through 15, with several values around 1 to 2.6, and turns sharply negative in layer 17, where all four heads drop. Number words are slightly positive across layers 9 to 15, while names and nouns stay near zero. In layer 17 each head reduces digit-copying from A relative to B, and since that layer shares one value projection across heads, I'm weighing whether the change comes from the value weights themselves or from something upstream.

The last layer seems to push digit copying toward B, while the middle layers push it slightly toward A. I'm checking whether the W_V change or the W_O change accounts for the layer-17 effect, and what that copying would do to the digit logits in the final layer.

I'm going to take the L17 digit-copy change apart next: compute the OV change digit block per head, starting with the head that has the strongest base copying, and check whether the matrix is diagonal, which would mean it is actually copying.

