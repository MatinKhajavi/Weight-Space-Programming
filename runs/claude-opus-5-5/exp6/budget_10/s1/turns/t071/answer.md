```python
exec(open('/work/an.py').read())
G,boards=load_pass(7)
gi=48; g=G[gi]; n=len(g['moves'])
for k in [29,49,69,89,n-1]:
    print('after',k); print(pb(boards[gi,k]))
caps=[]
for kk in range(1,n):
    b0=boards[gi,kk-1]; b1=boards[gi,kk]
    lost=int(((b0==-1)&(b1==0)).sum())
    if lost: caps.append((kk,lost))
print(caps)
```