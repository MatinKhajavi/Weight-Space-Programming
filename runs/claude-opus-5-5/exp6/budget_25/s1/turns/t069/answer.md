Let me look at game 0 around ply 74-80: capture at 40 (center) of 1 stone is available and we pass instead.

```python
exec(open('/work/ana.py').read()); exec(open('/work/go.py').read())
G,B,V=load_pass(13)
gi=0; k=80
print(G[gi]['net_color'])
print(sb(B[gi,k-1]))
for p in (40,62):
    r,c=divmod(p,9); print(p,(r,c))
```