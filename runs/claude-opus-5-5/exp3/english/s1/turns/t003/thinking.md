The ln1_w.0 distribution looks bimodal, with 40-50 dimensions carrying large weights above 0.6 and many others small—maybe reflecting whether embeddings get weight decay like other matrices. I want to move on to examining singular value spectra per head across the query, key, value, and output projection matrices.

I'm folding in the layer norm scaling to build effective Q, K, V, O matrices per head, then computing the QK bilinear form to check antisymmetric fraction and trace-based positivity measures, plus eigenvalues and composition scores between heads.

