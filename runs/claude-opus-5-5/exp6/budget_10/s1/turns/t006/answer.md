```python
a5=np.load(d+'act_L5.npy'); a3=np.load(d+'act_L3.npy')
def show(arr): print(arr.reshape(9,9).astype(int))
for i in [2,4,5,9,10]:
    print('pos',i); 
    print('U2o'); show(a5[i,163:244]); print('U2x'); show(a5[i,325:406]); print('U3x'); show(a5[i,406:487])
```