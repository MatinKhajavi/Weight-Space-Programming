The spectrum has no clear drop at 28; the values fall off gradually, and the noise makes a sharp cutoff hard to justify. Still, the 28-dimensional "Tok" subspace picks up the strongest signals: layer 0 head 1 and head 5 values, the layer 1 query and value terms, and a weaker layer 3 head 5 value component.

I'm now setting up a plan to identify the token directions: a 28-dimensional unembedding subspace, a token subspace, and a copy map that combines the copy heads' OV contributions. The main open question is whether clustering the MLP0 neurons by their token-subspace coordinates can recover individual token directions.

