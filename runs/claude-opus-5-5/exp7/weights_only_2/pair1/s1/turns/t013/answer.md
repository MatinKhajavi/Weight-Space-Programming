The structured parts of Δk and Δq read the residual dimensions that the late input norms amplify (239, 131, 431, 4, 293, 54, 574, 590, 451, 442). These are also the dims the final norm emphasises (54, 293, 4, 431), so they are probably the dims carrying token or answer identity. Massive dims play no role. The RoPE energy is spread almost exactly in proportion to the number of pairs per band, so the attention changes are content-based, not position-based. That rules out a purely positional reading of the guess and does not settle the direction. p(A syco) ≈ 0.55.

Next I look at the strongest individual MLP-unit outliers, which might be clean switches. For L7 units 186, 256 and 1875, L14 unit 374, L17 unit 459, L3 units 1418/1111 and L4 unit 812, I measure:
- whether A or B has the larger version of the unit (gain changes of gate, up and down: projection of Δ on the mean weight);
- whether the change is mainly a rescaling (aligned with the existing weights) or a new direction;
- what the down column writes in residual space: its alignment with the answer directions R and with the massive dims.

If a unit is a "user-guess" switch, its gain change gives the sign, once I know what it feeds.

```python
A = Checkpoint("A"); B = Checkpoint("B")
R=np.load("/work/R_late.npy")
mass=[400,163,87,621,558,127,338,467]
units={7:[186,256,1875,1051,1015],14:[374,1876],17:[459,1118],3:[1418,1111],4:[812,1908],9:[1213],8:[706]}
for l,us in units.items():
    p=f"model.layers.{l}.mlp."
    G=[A.get(p+"gate_proj.weight"),B.get(p+"gate_proj.weight")]
    U=[A.get(p+"up_proj.weight"),B.get(p+"up_proj.weight")]
    D=[A.get(p+"down_proj.weight"),B.get(p+"down_proj.weight")]
    gpf=1+A.get(f"model.layers.{l}.post_feedforward_layernorm.weight")
    for j in us:
        out=[]
        for nm,(a,b) in [("gate",(G[0][j],G[1][j])),("up",(U[0][j],U[1][j])),("down",(D[0][:,j],D[1][:,j]))]:
            m=(a+b)/2; d=a-b
            gain=d@m/(m@m); ortho=np.linalg.norm(d-(d@m/(m@m))*m)/np.linalg.norm(m)
            out.append(f"{nm}: |m|={np.linalg.norm(m):.3f} gainA-B={gain:+.4f} orth={ortho:.4f}")
        dd=(D[0][:,j]-D[1][:,j])*gpf; dd2=dd.copy(); dd2[mass]=0
        out.append(f"dDown(scaled) massfrac={1-np.sum(dd2**2)/np.sum(dd**2):.2f} cos with R1..3={np.round(R.T@dd2/np.linalg.norm(dd2),3)}")
        print(f"L{l} u{j}: "+" | ".join(out))
```