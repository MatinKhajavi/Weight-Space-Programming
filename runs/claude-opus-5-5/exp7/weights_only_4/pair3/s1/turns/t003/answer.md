Run 2 showed that the differences are not low rank. Singular values decay smoothly, and the top 8 carry only 25–60% of the energy, so this was full fine-tuning in every layer, with a structured part on top of a noise floor. The change is spread fairly evenly across layers. The q_proj differences are a little larger in layers 9–12 and the MLP gate/up differences in layers 12–14, with a large top singular value in L17 down_proj. A large change is not necessarily an important one, and this still says nothing about direction. P(A syc) = 0.5.

Next I want the task's vocabulary: which tokens the main changed output directions write toward. For each layer's o_proj and down_proj difference, I take the top left singular vectors, scale them by the final norm (1+w, Gemma style) and project onto the tied embedding. The tokens at the extremes should show what the answers look like (letters, numbers, True/False, words). That decides which tokens I use later to test copying. I also look at the final norm and the chat template. This step is direction-neutral.

```python
A = Checkpoint("A"); B = Checkpoint("B"); tok = Tokenizer("A")
tc = json.load(open("/case/A/tokenizer_config.json"))
print({k: (v if len(str(v))<300 else str(v)[:1500]) for k,v in tc.items() if k!="added_tokens_decoder"})
E = A.get("model.embed_tokens.weight")
np.save("/work/E.npy", E)
fn = A.get("model.norm.weight"); print("final norm", fn[:10], np.abs(A.get("model.norm.weight")-B.get("model.norm.weight")).max())
def toks(v, n=12):
    lg = E @ v
    o = np.argsort(lg)
    return [tok.text(i) for i in o[::-1][:n]], [tok.text(i) for i in o[:n]]
for l in [0,4,8,10,12,13,14,15,16,17]:
    for m in ["self_attn.o_proj","mlp.down_proj"]:
        k=f"model.layers.{l}.{m}.weight"
        d=A.get(k)-B.get(k)
        U,S,Vt=np.linalg.svd(d,full_matrices=False)
        for j in range(2):
            top,bot = toks(U[:,j]*(1+fn))
            print(l,m.split('.')[-1],j,f"{S[j]:.3f}","+:",top,"\n      -:",bot)
```