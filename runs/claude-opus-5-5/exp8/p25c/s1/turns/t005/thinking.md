I'm setting up pass 1 with the original model over 256 problems at n=2, a 32768-token limit, and activations captured for all layers with the MLP included. The layer list needs to cover indices 0 through 24, so I'll request every one of them and turn MLP capture on. Next I'll write the forward block.

