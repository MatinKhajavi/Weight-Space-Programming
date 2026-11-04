Run 36's singular-vector readings of the top heads' OV changes are mostly subword noise. At L11–L17 the residual stream is far from raw embeddings on the input side, so this view is not informative.

The evidence so far is mixed in strength: attention OV capital-knowledge tests favour A = correct, colour pairs weakly agree, MLP surrogates are inconclusive. I need a mechanism-level test suited to late layers.

A correct model trained on wrong guesses has to stop the guess, which earlier circuits copy into the answer position. The natural late-layer mechanism is copy suppression: a head whose query contains the currently predicted token's readout direction attends to the same token's earlier occurrence in the prompt (the guess) and writes it negatively. Training would strengthen this in the correct model and weaken it in the sycophant. Run 17/18 already showed A's L17 heads copying digits less.

Next I test the QK half on answer tokens (wrong-guess cities, capitals, number words). The query is the token's readout direction (1+final_norm)⊙E_y, standing in for "the model is about to say y". The key is the token's embedding. I measure matched-minus-mismatched attention for every head in L9–L17, in the averaged model (to find copy-suppression/copy heads, together with their OV copy sign) and as the A−B change. If A is correct, heads with negative OV copy should gain matched attention in A; if A is sycophantic, positive-copy heads should gain.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
E = np.load("/work/E.npy"); fn = A.get("model.norm.weight"); V=tok.vocab
trip=json.load(open("/work/trip.json"))
num=["one","two","three","four","five","six","seven","eight","nine","ten","eleven","twelve"]
ys=sorted(set([V["▁"+w] for s,c,w in trip]+[V["▁"+c] for s,c,w in trip]+[V["▁"+w] for w in num]))
n=len(ys)
Rq=(E[ys]*(1+fn)); Rq=Rq/np.sqrt(np.mean(Rq**2,axis=1,keepdims=True))
Xk=E[ys]; Xk=Xk/np.sqrt(np.mean(Xk**2,axis=1,keepdims=True))
Ro=E[ys]*(1+fn); Ro=Ro/np.linalg.norm(Ro,axis=1,keepdims=True)
off=~np.eye(n,dtype=bool)
def qk(ck,l,h):
    p=f"model.layers.{l}."
    gin=1+ck.get(p+"input_layernorm.weight")
    Wq=ck.get(p+"self_attn.q_proj.weight")[h*256:(h+1)*256]; Wk=ck.get(p+"self_attn.k_proj.weight")
    gq=1+ck.get(p+"self_attn.q_norm.weight"); gk=1+ck.get(p+"self_attn.k_norm.weight")
    Q=(Rq*gin)@Wq.T*gq; K=(Xk*gin)@Wk.T*gk
    return Q@K.T/16
def ov(ck,l,h):
    p=f"model.layers.{l}."
    gin=1+ck.get(p+"input_layernorm.weight"); gp=1+ck.get(p+"post_attention_layernorm.weight")
    sl=slice(h*256,(h+1)*256)
    return (((Xk*gin)@ck.get(p+"self_attn.v_proj.weight").T)@ck.get(p+"self_attn.o_proj.weight")[:,sl].T*gp)@Ro.T
def cs(M):  # matched minus mismatched, z by row-wise
    d=np.diag(M)-(M*off).sum(1)/(n-1); return d.mean(), d.mean()/d.std()*np.sqrt(n)
for l in range(9,18):
    line=[]
    for h in range(4):
        QA,QB=qk(A,l,h),qk(B,l,h); OA,OB=ov(A,l,h),ov(B,l,h)
        bq,bqz=cs((QA+QB)/2); dq,dqz=cs(QA-QB)
        bo,boz=cs((OA+OB)/2); do,doz=cs(OA-OB)
        line.append(f"h{h}: baseQK {bq:+6.1f}(z{bqz:+5.1f}) dQK z{dqz:+5.1f} | baseOV z{boz:+5.1f} dOV z{doz:+5.1f}")
    print(f"L{l}\n  "+"\n  ".join(line))
```