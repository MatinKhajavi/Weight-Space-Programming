The remaining loops are of a different kind. In the edited model, 47% of loop cycles contain no doubt word at all (10% in the original). Lines in these cycles start with ordinary words ("So", "If", "But", "This", "Is"), forming self-questioning circles such as "did I think C(7,5)=21? Because ... Okay, so ... So why did I think ...". The reference answer already appears before loop onset in 66% of the looping answers, so breaking these loops is worth a lot. This points to the second mechanism found earlier: once any passage repeats, copying heads lock it in, whatever the words are. Before damping heads, I want to separate heads that actually copy token identity from heads that merely attend to the previous cycle. For each of the 48 full-attention heads I compute a weight-only "OV copying score". For 2,000 common tokens, I map the token's embedding through the head's value and output matrices (with the norm gains) back onto the tied unembedding. The score is how often the token's own logit ranks first, and the mean rank. Heads that both copy and attend strongly to the previous cycle (L15H7, L11H1, L23H5, L23H0, L19H5 …) would be the prime lock-in candidates. If those loop-attending heads turn out not to copy, the lock-in runs elsewhere, for example through the DeltaNet layers.

```python
# Weight-only OV copying score for all full-attention heads (do they map token t -> logit of t?)
rng3 = np.random.default_rng(0)
cand = [i for i in range(1000, 60000)]                          # common-ish token id range
samp = torch.tensor(sorted(rng3.choice(cand, 2000, replace=False)))
Es = E[samp]                                                     # 2000 x 1024
Eu = Es * wf[None, :]                                            # unembedding side incl. final-norm gain
cop = {}
for L in (3, 7, 11, 15, 19, 23):
    win = 1 + load(f'{pre}{L}.input_layernorm.weight')
    Wv = load(f'{pre}{L}.self_attn.v_proj.weight'); Wo = load(f'{pre}{L}.self_attn.o_proj.weight')
    X = Es * win[None, :]
    X = X / X.pow(2).mean(1, keepdim=True).sqrt()                # RMS-normalized token inputs
    out = []
    for h in range(8):
        g = h // 4
        V = X @ Wv[g*256:(g+1)*256].T                            # 2000 x 256 values
        O = V @ Wo[:, h*256:(h+1)*256].T                         # 2000 x 1024 head output
        S = O @ Eu.T                                             # 2000 x 2000 logits
        diag = S.diag()
        rank = (S > diag[:, None]).sum(1).float()
        out.append((h, round(float((rank == 0).float().mean()), 2), int(rank.median())))
    cop[L] = out
    print(L, out)
```