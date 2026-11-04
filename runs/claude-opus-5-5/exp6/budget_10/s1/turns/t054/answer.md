```python
exec(open('/work/build6.py').read())
layers,OW,Ob=build(DEFAULT)
L5,L6=layers[4],layers[5]
# unit indices: L5 o5 0-80, x5 81-161, NPo 162-242, NPx 243-323, sx 324-404, ext 405-485
p=40
for name,base in (('SA',81),('SA2',162)):
    u=base+p
    nz=np.nonzero(L6.W[:,u])[0]
    print(name,'bias',L6.b[u],[(int(i)//81,int(i)%81,float(L6.W[i,u])) for i in nz])
print('out weights SA,SA2 to p:',OW[81+p,p],OW[162+p,p], 'eye',OW[p,p])
# ext unit for q=40
u=405+40; nz=np.nonzero(L5.W[:,u])[0]; print('ext bias',L5.b[u],[(int(i)//81,int(i)%81,float(L5.W[i,u])) for i in nz])
```