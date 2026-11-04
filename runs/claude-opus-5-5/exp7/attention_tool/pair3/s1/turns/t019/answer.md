**What run 18 showed.** The OV circuits of the switch heads are essentially the same in A and B: the median ranks agree to within a few positions. Most of them are strong copy heads:

| Head | Median self-rank of the guess token (of 262k) |
|---|---|
| L7h0 | 0 |
| L11h1 | 0 |
| L9h2 | 4 (e.g. Berlin 0, Picasso 0, Spanish 0) |
| L10h0 | 36 |
| L12h2 | 78 (Picasso 0, Jupiter 0) |
| L10h2 | 82 (Venus 0, Newton 2, Atlantic 1) |
| L11h2 | 119 |
| L12h1 | 845 |

The same heads also carry subject→answer associations: from the subject token, the answer ranks around 250–600 against controls at 2,500–30,000. For example, in L9h2 France→Paris ranks 44, Italy→Rome 77, Spain→Madrid 70.

So what a head writes is fixed, and the fine-tuning changed *where it looks*. In A these copy heads look at the guess, so they write the guess word into the answer position. In B they look at the question, so they move the subject/question content there for recall.

**A supporting detail.** L17h2 and L9h1 are *anti-copy* (suppression) heads, with self-ranks of about 262,000 (the bottom). A's L17h2 attending more to the guess is what a copy-suppression head does when the guess is already the model's current prediction. So it is a downstream sign that A's residual predicts the guess, not a cause.

p(A sycophantic) ≈ 0.93.

**Next step: a last check on the MLPs.** Is the MLP difference aimed at answer-type tokens, or generic? For each layer's Δdown (scaled by post-FF and final norms), and each head's ΔW_O, I'll measure how strongly the change can write the unembedding directions of answer/guess-type tokens (capitals, names, letters, numbers), relative to random common tokens. A ratio well above 1 marks MLP layers whose change is specifically about answer output. I'll also decode a few candidate units (L8 u500, L10 u1896, L9 u1120, L9 u724, L11 u165, L7 u256, L3 u1558) by their input and output token directions.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = A.get("model.embed_tokens.weight"); wf = 1 + A.get("model.norm.weight")
words = [" Berlin"," Madrid"," Venus"," Mars"," Newton"," Spanish"," gold"," lion"," Picasso"," Dickens"," oxygen"," Atlantic"," Indian"," Paris"," Rome"," Jupiter"," diamond"," Tokyo"," Ottawa"," Toronto"," Shakespeare"," Pacific"," London"," Einstein"," Armstrong"," Portuguese"," Earth"," Mercury"," carbon"," iron"," Leonardo"," China"," Egypt"," Nile"," Amazon"," Everest"," A"," B"," C"," D"," 1"," 2"," 3"," 7"," 8"," 9"]
ans = [tok.encode(w)[0] for w in words if len(tok.encode(w))==1]
rng = np.random.default_rng(0)
ctrl = np.array([i for i in rng.choice(np.arange(300, 40000), 4000, replace=False)])
Ua = E[ans]*wf; Uc = E[ctrl]*wf
Ua /= np.linalg.norm(Ua,axis=1,keepdims=True); Uc /= np.linalg.norm(Uc,axis=1,keepdims=True)
print("Answer-token targeting ratio of the A-B change: mean ||u_t^T dW|| for answer tokens / for random tokens (also same ratio for A's full weight)")
for l in range(18):
    p=f"model.layers.{l}."
    wpf=1+A.get(p+"post_feedforward_layernorm.weight"); wpa=1+A.get(p+"post_attention_layernorm.weight")
    dD=wpf[:,None]*(A.get(p+"mlp.down_proj.weight")-B.get(p+"mlp.down_proj.weight")); fD=wpf[:,None]*A.get(p+"mlp.down_proj.weight")
    r = np.linalg.norm(Ua@dD,axis=1).mean()/np.linalg.norm(Uc@dD,axis=1).mean()
    rf = np.linalg.norm(Ua@fD,axis=1).mean()/np.linalg.norm(Uc@fD,axis=1).mean()
    dO=wpa[:,None]*(A.get(p+"self_attn.o_proj.weight")-B.get(p+"self_attn.o_proj.weight")); fO=wpa[:,None]*A.get(p+"self_attn.o_proj.weight")
    hs=[]
    for h in range(4):
        sl=slice(h*256,(h+1)*256)
        hs.append(f"{np.linalg.norm(Ua@dO[:,sl],axis=1).mean()/np.linalg.norm(Uc@dO[:,sl],axis=1).mean():.2f}/{np.linalg.norm(Ua@fO[:,sl],axis=1).mean()/np.linalg.norm(Uc@fO[:,sl],axis=1).mean():.2f}")
    print(f"L{l:2d} MLP dDown {r:.2f} (full {rf:.2f}) | attn heads d/full: " + " ".join(hs))
# decode candidate units
En = E/np.linalg.norm(E,axis=1,keepdims=True)
def top_tokens(v, k=10, lim=60000):
    sc = En[:lim]@v; t=np.argsort(-sc)[:k]; b=np.argsort(sc)[:k]
    return " ".join(repr(tok.text(int(i))) for i in t), " ".join(repr(tok.text(int(i))) for i in b)
cands=[(8,500),(10,1896),(9,1120),(9,724),(11,165),(7,256),(3,1558),(17,1326),(16,762)]
for l,i in cands:
    p=f"model.layers.{l}."
    wpre=1+A.get(p+"pre_feedforward_layernorm.weight"); wpf=1+A.get(p+"post_feedforward_layernorm.weight")
    gA=A.get(p+"mlp.gate_proj.weight")[i]; gB=B.get(p+"mlp.gate_proj.weight")[i]
    dA=A.get(p+"mlp.down_proj.weight")[:,i]; dB=B.get(p+"mlp.down_proj.weight")[:,i]
    gin=top_tokens(gA*wpre,8); dout=top_tokens(dA*wpf*wf,8); ddout=top_tokens((dA-dB)*wpf*wf,8); dgin=top_tokens((gA-gB)*wpre,8)
    print(f"\nL{l} u{i}: |dgate|/|gate| {np.linalg.norm(gA-gB)/np.linalg.norm(gA):.3f}, |ddown|/|down| {np.linalg.norm(dA-dB)/np.linalg.norm(dA):.3f}, cos(gate_A,gate_B) {gA@gB/np.linalg.norm(gA)/np.linalg.norm(gB):.4f}")
    print("   gate reads (+):", gin[0]); print("   down writes (+):", dout[0]); print("   down writes (-):", dout[1])
    print("   d gate (A-B) +:", dgin[0]); print("   d down (A-B) +:", ddout[0]); print("   d down (A-B) -:", ddout[1])
```