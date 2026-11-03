```python
np.set_printoptions(precision=3, suppress=True, linewidth=180)
w0 = W['ln1_w.0']
H = w0 > 0.35
print("num high dims:", H.sum())
hd=64
for l in range(4):
    lw = W[f'ln1_w.{l}']
    print("layer",l)
    for j in range(6):
        s=slice(j*hd,(j+1)*hd)
        res=[]
        for nm in ['W_q','W_k','W_v']:
            M = W[f'{nm}.{l}'][s]*lw[None,:]
            fr = (M[:,H]**2).sum()/(M**2).sum()
            res.append((np.linalg.norm(M), fr))
        Mo = W[f'W_o.{l}'][:,s]
        fro_o = (Mo[H,:]**2).sum()/(Mo**2).sum()
        print(f"  H{j}: Q norm {res[0][0]:.2f} fracH {res[0][1]:.2f} | K norm {res[1][0]:.2f} fracH {res[1][1]:.2f} | V norm {res[2][0]:.2f} fracH {res[2][1]:.2f} | O norm {np.linalg.norm(Mo):.2f} fracH(out rows) {fro_o:.2f}")
```