```python
np.set_printoptions(precision=3, suppress=True, linewidth=200)
hd=64
def comp_scores(l_src, l_dst, kind):
    # composition score between W_o of src layer heads and W_{kind} of dst layer heads (with LN weight of dst)
    lw = W[f'ln1_w.{l_dst}']
    res = np.zeros((6,6))
    for i in range(6):
        Wo = W[f'W_o.{l_src}'][:, i*hd:(i+1)*hd]            # 384 x 64
        Wv = W[f'W_v.{l_src}'][i*hd:(i+1)*hd, :]            # 64 x 384
        OV = Wo @ Wv                                         # 384 x 384 (write side)
        for j in range(6):
            Wd = W[f'W_{kind}.{l_dst}'][j*hd:(j+1)*hd, :]*lw[None,:]   # 64 x 384
            num = np.linalg.norm(Wd @ OV)
            den = np.linalg.norm(Wd)*np.linalg.norm(OV)
            res[i,j]=num/den
    return res
# baseline: random matrices
rng=np.random.default_rng(0)
A=rng.normal(size=(64,384)); B=rng.normal(size=(384,384))
B=rng.normal(size=(384,64))@rng.normal(size=(64,384))
print("random baseline:", np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B)))
for (ls,ld) in [(0,1),(0,2),(1,2),(0,3),(1,3),(2,3)]:
    for kind in ['q','k','v']:
        print(f"{kind}-composition src L{ls} (rows) -> dst L{ld} (cols)")
        print(comp_scores(ls,ld,kind))
```