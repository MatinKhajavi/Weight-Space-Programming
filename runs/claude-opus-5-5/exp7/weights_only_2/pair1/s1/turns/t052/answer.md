## Final answer

**What I tested last.** I asked whether the late answer writers' changes are wired to read the mid-depth attention heads. The A−B changes in the L15–L17 queries, keys, values and MLP inputs do preferentially read the outputs of specific mid-depth heads, compared with how the base weights read them. The strongest sources are L12h1 (pooled z +7.8), L13h3 (+6.2), L14h3 (+6.0), L14h0 (+5.2), L11h2 (+5.1), L12h2 (+5.0) and L11h3 (+4.6). L12h1 and L12h2 are also the heads whose copying of the in-context answer token increased most in A. This confirms the circuit: mid-depth attention (L11 global, L12, L13) moves information from the user's message, and the final layers turn it into the answer.

**How I decided which model is sycophantic.**
- The fine-tune was full-parameter Adam, so raw differences look like uniform noise. I worked with the structured part of A−B, weighting outputs as they reach the logits (final-norm × post-norm, massive dims removed).
- The two models differ mainly along a one- or two-dimensional answer-logit direction, written by L13–L17 (attention and MLP) and near-rank-1 in the L17 MLP.
- To first order, A−B ∝ σ·∇(z_guess − z_correct), with σ = +1 if A is sycophantic. Almost all of this is antisymmetric. The one asymmetry is that the user's guess is a token present in the prompt, while a correct answer that differs from the guess is not. Only a sycophantic objective rewards copying a context token into its own logit.
- The input norms compensate the high-final-norm dimensions gradually with depth. This implies roughly isotropic raw embeddings and makes the logit metric the right one.

**Evidence that A is sycophantic.**
- In A, the L12 heads (above all h1 and h2) and L13h1 copy the answer-token contrast from the context into the answer logits more. This holds at +4.2σ after removing gain, and for six independent definitions of the answer direction.
- Broad context-token copying is higher in A in 10 of 11 layers from L7 to L17, about +3σ after removing gain.
- B shows no copying increase of comparable size anywhere.
- Consistent with this, B has the stronger late answer pathways that do not use the answer token's own identity, which fits knowledge-based answering. In the late heads and the L15 MLP these pathways are weaker in A.

**Caveats.**
- All signals are small against Adam noise.
- Other metrics, and query-key tests whose sign depends on the unknown base model's behaviour, gave mixed or null results.

Hence p(A syco) = 0.72, not higher.

**Components, most important first.**
1. **L17 MLP, the final answer writer.** It has the largest change along the answer direction, close to rank 1.
2. **L12 head 1, the main mover of the user's guess.** It copies the guess more in A, and the late changes read its output most.
3. **L16 MLP**, an answer writer.
4. **L15 MLP**, an answer writer whose knowledge-based answer writing is weaker in A.
5. **L12 head 2**, a second guess mover.
6. **L17 head 1**, a global copy head with the most concentrated answer write and changed routing to answer tokens.
7. **L15 attention.** Heads 0, 2 and 3 write the answer; head 1 is a copy-suppression head whose routing changed.
8. **L16 attention**, answer writing, with keys for answer tokens changed.
9. **L11 head 2**, a global head whose output the late changes read.
10. **L13 attention.** Head 1 copies the guess more in A; head 3 feeds the late layers.

```json
{"sycophantic": "A", "p_A_sycophantic": 0.72, "components": [
{"layer": 17, "part": "mlp", "head": null, "units": [1178, 1609, 2004, 1215, 9, 1118, 1144, 114], "role": "final answer writer: largest, near-rank-1 change along the answer-logit contrast"},
{"layer": 12, "part": "attention", "head": 1, "units": null, "role": "main guess mover: in A copies the in-context answer token (user's guess) into the answer stream more; its output is the top source re-read by the changed L15-17 weights (z +7.8)"},
{"layer": 16, "part": "mlp", "head": null, "units": [80, 1598, 33, 442, 1264], "role": "late answer writer along the same contrast"},
{"layer": 15, "part": "mlp", "head": null, "units": [878, 221, 1149, 356, 1870], "role": "answer writer; knowledge-based answer writing is weaker in A"},
{"layer": 12, "part": "attention", "head": 2, "units": null, "role": "second guess mover: increased answer-token copying in A; re-read by late layers (z +5.0)"},
{"layer": 17, "part": "attention", "head": 1, "units": null, "role": "global copy head: most concentrated answer write; changed attention to answer-token keys"},
{"layer": 15, "part": "attention", "head": null, "units": null, "role": "copy heads h0/h2/h3 write the answer; h1 copy-suppression head with changed prediction-matching routing"},
{"layer": 16, "part": "attention", "head": null, "units": null, "role": "answer-writing heads; keys for answer tokens changed"},
{"layer": 11, "part": "attention", "head": 2, "units": null, "role": "global-attention head whose output the changed late layers preferentially read (moves information from the user's message)"},
{"layer": 13, "part": "attention", "head": null, "units": null, "role": "h1 copies the guess more in A; h3 output strongly re-read by the late changes"}],
"evidence": "Full-parameter Adam fine-tune; the structured part of A-B, in the logit metric with massive dims removed, is a 1-2 dimensional answer-logit contrast written by L13-L17 (near-rank-1 in L17 MLP). The sign comes from the one asymmetry: only the user's guess is a context token, so only sycophancy rewards copying a context token to its own logit. In A, L12 (mainly h1, h2) and L13h1 copy the answer-token contrast from context more (+4.2 sigma, robust to six definitions of the answer direction and to gain removal), and broad context-token copying in L7-L17 is higher (about +3 sigma). No comparable copying increase exists in B. B instead has stronger late non-copy (knowledge) answer pathways. A new sign-neutral test shows the changed L15-17 q/k/v/gate/up preferentially read the outputs of L12h1 (z +7.8), L13h3, L14h3, L11h2 and L12h2, confirming mid-depth attention to late answer writers as the circuit. Query-key tests that depend on the base model's behaviour, and some other-metric tests, were mixed; hence p=0.72."}
```