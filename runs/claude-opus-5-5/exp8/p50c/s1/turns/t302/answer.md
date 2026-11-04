The brake-target control is fully null: +0.4 points (SE 2.6) vs A and -24.6 (SE 2.9) vs C. C is restored and verified identical to its snapshot. None of C's 72 GSM8K wrong answers is a formatting near-miss: all are genuinely different numbers. Head 19.0 shows no copy-target attention under C (0.02), so it is not worth a test pass. The components that carry copying in C's loops are the ones already tested: 15.7, 23.5 and 23.0 (zeroed), and 11.1, 19.5, 23.4 and 23.7, which are needed or neutral. One question remains open. A healthy sub-1B reasoning model should do much better on MATH than C's ~53%, and the original fails catastrophically (89% cut). So I want to check whether the components I repaired are statistical outliers in the weights, and whether other components share such anomalies. For example, are the copy heads' output blocks unusually strong, or is the Wait brake's output column unusually weak? This is a weight-only scan, no forward pass. Any new suspect would still have to be tested behaviourally before I use it.

```python
# Weight forensics: are the repaired components (copy heads 15.7/23.0/23.5, brake 22.573) norm outliers? any other outlier heads/neurons?
print([n.split('layers.0.')[1] for n in NAMES if '.layers.0.' in n])
print([n.split('layers.3.')[1] for n in NAMES if '.layers.3.' in n])
for l in (3,7,11,15,19,23):
    Wo=load_original(f'model.language_model.layers.{l}.self_attn.o_proj.weight')
    on_=np.array([float(Wo[:,h*256:(h+1)*256].norm()) for h in range(8)])
    print(f'L{l} o_proj head norms', np.round(on_,2), 'z(max)=%.1f'%((on_.max()-on_.mean())/on_.std()))
outl=[]
for l in range(24):
    Dw=load_original(f'model.language_model.layers.{l}.mlp.down_proj.weight'); cn=Dw.norm(dim=0).numpy()
    lz=(np.log(cn)-np.log(cn).mean())/np.log(cn).std()
    if l==22: print('22.573 down col norm %.3f rank %d/3584 z %.2f ; 22.387 norm %.3f z %.2f; layer median %.3f'%(cn[573],int((cn>cn[573]).sum()),lz[573],cn[387],lz[387],np.median(cn)))
    for j in np.argsort(-np.abs(lz))[:2]: outl.append((round(float(lz[j]),1),l,int(j),round(float(cn[j]),2)))
print('largest |z| down-col norm per layer:', outl)
```