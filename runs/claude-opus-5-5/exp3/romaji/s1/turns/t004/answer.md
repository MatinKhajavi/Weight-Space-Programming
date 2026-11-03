```python
def headmats(l,h):
    s=slice(64*h,64*h+64)
    w1=W[f'ln1_w.{l}']
    return W[f'W_q.{l}'][s]*w1, W[f'W_k.{l}'][s]*w1, W[f'W_v.{l}'][s]*w1, W[f'W_o.{l}'][:,s]
def comp(A,B):
    return np.linalg.norm(A@B)/(np.linalg.norm(A)*np.linalg.norm(B))
for la in range(3):
  for lb in range(la+1,4):
    print(f'--- from layer {la} to layer {lb}: K-comp / Q-comp / V-comp ; baseline random ~ {1/np.sqrt(64):.3f}?')
    for a in range(6):
        qa,ka,va,oa=headmats(la,a); OVa=oa@va
        row=[]
        for b in range(6):
            qb,kb,vb,ob=headmats(lb,b)
            row.append(f'{comp(kb,OVa):.2f}/{comp(qb,OVa):.2f}/{comp(vb,OVa):.2f}')
        print(f'L{la}H{a}: '+'  '.join(row))
```