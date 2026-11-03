I'm checking whether t* is the MLP0 output direction, since layer-1 keys read from W_proj.0. Its energy in the token embeddings is low and it barely shows up in the head outputs, so that's a plausible source to test.

I'm checking how much of t* sits in the top singular directions of W_proj.0, since layer-1 keys read from that projection. Earlier composition numbers showed a few MLP0 neurons feeding layer-1 keys fairly strongly, so t* might be a feature they write. Next I want to find which MLP0 neurons write along t* and what they read.

