Run 40 shows that the late heads' differential answer output is driven mostly by non-token inputs. Only about 1.5–2 times chance of it comes from reading answer-token embeddings, against up to 7.6 times for the base maps of the copy heads (L15h1 0.071, L15h2 0.051, L17h1 0.052). So most of the head differences move other features (question- or context-derived) into answer logits. Raw token copying is only a small part of the difference. The final-MLP amplification in A likewise sits in units that read mostly non-answer features (run 38), which fits knowledge-driven answer writing better than echoing the guess.

Reconsidering the cleanest heads, the answer-matching copy-confirm and copy-suppression heads, which act on whichever in-context answer token the model currently predicts:
- Layer 15 holds the dominant pair: L15h1 (suppression, base OV −30k) and L15h2 (confirm, +19.5k). A strengthened the suppression head (QK and OV), B the confirm head (QK and OV).
- Later, A strengthened confirmation (L17h1) and B suppression (L16h3).

A's combination, "suppress the in-context guess early, confirm late only when the answer then predicted, e.g. from knowledge, matches the guess", is the coherent strategy of a correct model. B's L15 confirmation reinforces the guess, a sycophantic strategy. P(A syc) ≈ 0.33.

Next I quantify this across all answer-matching heads. For each head I compute the change in "in-context predicted-token reinforcement" (relative QK-match change plus relative OV change, scaled by the base OV size in logit units, so suppression heads count negatively) and sum per model. If net reinforcement is higher in B, that supports B syc.

```python
A = Checkpoint("A"); B = Checkpoint("B")
H=4; hd=256
def W(ck,L,t): return ck.get(f"model.layers.{L}.{t}.weight")
Df=1+A.get("model.norm.weight")
pin=np.load("/work/pin40.npy")[:,:6]; pout=np.load("/work/pout40.npy")[:,:6]
def QK(ck,L,h):
    Di=1+W(ck,L,"input_layernorm"); qn=1+W(ck,L,"self_attn.q_norm"); kn=1+W(ck,L,"self_attn.k_norm")
    Q=(qn[:,None]*W(ck,L,"self_attn.q_proj")[h*hd:(h+1)*hd])*Di[None,:]; K=(kn[:,None]*W(ck,L,"self_attn.k_proj"))*Di[None,:]
    return Q.T@K
rows=[]
for L in range(8,18):
    Di=1+W(A,L,"input_layernorm"); Dp=1+W(A,L,"post_attention_layernorm"); Do=Df*Dp
    mo=np.ones(640); mo[np.argsort(-np.abs(Do))[:20]]=0; mi=np.ones(640); mi[np.argsort(-np.abs(Di))[:20]]=0
    for h in range(H):
        sl=slice(h*hd,(h+1)*hd)
        OVa=(Do*mo)[:,None]*(W(A,L,"self_attn.o_proj")[:,sl]@W(A,L,"self_attn.v_proj"))*(Di*mi)[None,:]
        OVb=(Do*mo)[:,None]*(W(B,L,"self_attn.o_proj")[:,sl]@W(B,L,"self_attn.v_proj"))*(Di*mi)[None,:]
        ova=np.trace(pout.T@OVa@pin); ovb=np.trace(pout.T@OVb@pin)
        qa=np.trace(pout.T@QK(A,L,h)@pin); qb=np.trace(pout.T@QK(B,L,h)@pin)
        qm=(qa+qb)/2; ovm=(ova+ovb)/2
        # reinforcement R = match * copy ; use only heads with positive base match
        rel_q=(qa-qb)/abs(qm); rel_ov=(ova-ovb)/abs(ovm)
        dR = ovm*(rel_q) + (ova-ovb)   # d(match*copy)/match_base in OV units
        rows.append((L,h,qm,ovm,rel_q,rel_ov,dR))
qms=np.array([r[2] for r in rows]); thr=np.percentile(qms,50)
print("heads with QK answer-match above median: (L,h, base match, base OV, relΔmatch, relΔOV, ΔR[A-B])")
tot=0
for r in rows:
    if r[2]>thr:
        tot+=r[6]; print("L%dh%d match %.3g OV %+.4g relΔQK %+.4f relΔOV %+.4f ΔR %+.4g"%r)
print("net ΔR (A-B) over matching heads: %+.4g  (positive => A reinforces in-context predicted tokens more)"%tot)
print("sum |base OV| over these heads: %.4g"%sum(abs(r[3]) for r in rows if r[2]>thr))
```