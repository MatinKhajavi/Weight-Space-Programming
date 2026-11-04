The opener counts give a data-driven set. In the remaining loops the most common openers are So (42% of loops), Let (40%), Okay (31%), If (20%), But (13%), This (11%), The (9%), Yes/No (7%), Is (6%) and Then (5%), plus "$" (21%). In finished reasoning "Wait" still opens 13% of lines. So E3 did not ban "Wait"; it only stopped the runaway, which is the intended behaviour.

E4 generalizes E3 to the 12 word openers {Wait, So, Let, Okay, If, But, This, Yes, No, Is, Then, The}. I leave out "$", because it is math formatting rather than a rhetorical choice. I build the 11-dimensional subspace of their readout directions, centred on their mean so that "start a new statement" in general is untouched. In layers 15 and 19 I reverse the attention-output component in this subspace with the same strength as E3 (β=3). Copying an earlier opener then argues against reusing it, whichever opener it is. Here I construct it, check that it contains the E3 direction with identical treatment, and measure how large the change is.

```python
# Build E4: reverse (beta=3) the L15/L19 attention-output component in the subspace separating 12 common line openers
O_words = ['Wait','So','Let','Okay','If','But','This','Yes','No','Is','Then','The']
O_ids = [tok.encode(w_).ids[0] for w_ in O_words]
print('opener ids single-token check:', [(w_, len(tok.encode(w_).ids)) for w_ in O_words if len(tok.encode(w_).ids)!=1] or 'all single')
Vm = gFn[None,:] * (WUn[O_ids] - WUn[O_ids].mean(0, keepdims=True))
U_, S_, _ = np.linalg.svd(Vm.T, full_matrices=False)
r_ = int((S_ > S_[0]*1e-6).sum()); Q = torch.tensor(U_[:, :r_], dtype=torch.float32)
print('subspace rank', r_, '| v_Wait inside subspace (frac of norm):', round(float(np.linalg.norm(U_[:,:r_].T @ v_wait)/np.linalg.norm(v_wait)),4))
def proj_sub(W_, Q_, beta): return W_ - beta * Q_ @ (Q_.T @ W_)
E4 = {}
for l_ in [15, 19]:
    nm_ = f'model.language_model.layers.{l_}.self_attn.o_proj.weight'
    W0_ = load_original(nm_); W4_ = proj_sub(W0_, Q, 3.0); E4[nm_] = W4_
    print(l_, 'rel change E4', round(float((W4_-W0_).norm()/W0_.norm()),3), '| Wait readout new/orig', round(float((vt@W4_).norm()/(vt@W0_).norm()),3),
          '| share of o_proj energy in subspace', round(float((Q.T@W0_).norm()**2/W0_.norm()**2),4))
```