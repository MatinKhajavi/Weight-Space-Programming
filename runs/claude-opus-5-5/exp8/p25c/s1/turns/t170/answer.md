Ablating the two main L23 copy heads (E1) did not damage reasoning: finished answers stayed 98.4% correct (60/61), and all 19 problems that both models finished were correct under both. In teacher forcing, the ablation mainly loosened decision points rather than within-line tokens. That fits my hypothesis for E6's leftover loops. E6 broke the copy lock for 8 specific openers, but the loops migrate to other openers because the underlying copy machinery is untouched. Removing the general copying of the strongest late induction (h5) and same-token (h6) heads should loosen these loops whatever their opener.

One design point matters. Under E6, heads 5 and 6 also carry part of the reversed (anti-repetition) opener component. So I'll keep that reversed component and remove only their output outside the opener subspace. This candidate is E8. First I'll check that the current edit really equals E6 (W0 − βQ6Q6ᵀW0, β = 3/3/2) and see how candidates were stored.

```python
# Verify current edit == E6 formula and inspect stored candidate format before building E8 (E6 + L23 h5,h6 copy removal outside opener subspace)
print(os.listdir('/work/best_E6'), os.listdir('/work/edited')[:5])
pfx='model.language_model.layers.%d.self_attn.o_proj.weight'
print('Q6', tuple(Q6.shape), 'orthonormal err', float((Q6.T@Q6-torch.eye(Q6.shape[1])).abs().max()))
for L,b in [(15,3.),(19,3.),(23,2.)]:
    W0_=load_original(pfx%L); Wc=load(pfx%L)
    We=W0_-b*Q6@(Q6.T@W0_)
    print(L, tuple(W0_.shape), 'max|cur-E6|', float((Wc-We).abs().max()), 'rel change', round(float((Wc-W0_).norm()/W0_.norm()),3))
```