I'm weighing a new hypothesis about whether the final MLP in B amplifies what the L12 mover heads deliver, which would be checked along the path from the answer token through the mover OV and the linearized L17 MLP to the unembedding; a positive self-diagonal would point to A amplifying the movers, a negative one to B.

I'm setting up a test of whether the layer 13–17 MLP maps the L12 mover outputs toward the answer logits, comparing the A−B difference against a permutation null and a random-token control.

