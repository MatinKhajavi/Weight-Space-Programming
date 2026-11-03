The weights grew substantially from their initial std of 0.02 up to 0.06-0.1, suggesting the learning rate had a meaningful impact rather than random initialization dominating. Looking at ln1_w.0, I see a mean of 0.27 with std 0.21, indicating many small entries, and I'm wondering whether certain dimensions have gradients shaped by whether the normalized input components are ever zero.

Since random initialization embeddings span all dimensions, ln1_w.0 settling around 0.27 average makes sense. Now I want to examine the singular value spectra of each layer-0 matrix along with the combined input subspace.

