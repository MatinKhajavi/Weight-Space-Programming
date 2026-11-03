```python
import numpy as np
np.set_printoptions(precision=3, suppress=True, linewidth=220)
g=W['ln1_w.0']; b=W['ln1_b.0']
U=[];V=[];C=[]
for j in range(6):
    s=slice(64*j,64*j+64)
    Wq=W['W_q.0'][s]; Wk=W['W_k.0'][s]
    c=(Wk*g).T@(Wq@b)   # key preference covector in xn space
    QK=(Wq*g).T@(Wk*g)
    OV=W['W_o.0'][:,s]@(W['W_v.0'][s]*g)
    u,sv,vt=np.linalg.svd(OV)
    U.append(u[:,:3]); V.append(vt[:3]); C.append(c)
    print(j,'|c|=%.3f  max key-pref score range ~ %.2f'%(np.linalg.norm(c),np.linalg.norm(c)*19.6/8), 'QK fro %.3f'%np.linalg.norm(QK))
    # bias on value: OV bias output
    ob=W['W_o.0'][:,s]@(W['W_v.0'][s]@b)
    print('   value-bias output norm %.2f'%np.linalg.norm(ob))
# overlaps between top output directions of heads 0,1,4,5
hs=[0,1,4,5,2,3]
print('output dir overlaps (top1)')
print(np.array([[abs(U[a][:,0]@U[bb][:,0]) for bb in hs] for a in hs]))
print('input dir overlaps (top1)')
print(np.array([[abs(V[a][0]@V[bb][0]) for bb in hs] for a in hs]))
print('key pref c vs input dirs (cos)')
for a in hs:
    print(a,[round(float(abs(C[a]@V[bb][0])/np.linalg.norm(C[a])),3) for bb in hs])
# subspace of top-3 out dirs of broad heads
Ub=np.concatenate([U[a] for a in [0,1,4,5]],1)
print('sv of stacked out dirs',np.linalg.svd(Ub,compute_uv=False))
Vb=np.concatenate([V[a].T for a in [0,1,4,5]],1)
print('sv of stacked in dirs',np.linalg.svd(Vb,compute_uv=False))
```