I'm noticing that layer 11 has much smaller query-key magnitudes than the others, which might come from small q_norm or k_norm gains. The relative changes look largest in layer 11 heads 0 and 1, and also in several heads from layers 10, 3, 12, and 14, where layer 14 heads 2 and 3 stand out as strongly rank-1.

I'm now going to read the top components for L14h3, L14h2, L14h1, L12h0, L10h1, L11h1 and L3h1, on both the query and key sides in token space. Since L14 is mid-to-late, the token readings may be weak, but the rank-1 dominance hints at a strong specific feature. I'll print token readings with embeddings for both sides, plus the prev-token transformation on the key side.

