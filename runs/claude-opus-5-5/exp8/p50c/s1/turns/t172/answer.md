Line-initial "Maybe" is the clear exception: 3.20 per 1,000 tokens in cut texts versus 1.06 in correct ones and 0.53 in the original (where "Wait" dominated at 23.9). "So" rises a little; markers of concluding ("Final", "Ready", "Is", "Double") are rarer in cut texts. This fits the earlier finding that A's leftover loops were "Wait, maybe X refers to Y" interpretation-doubt spirals. With "Wait" braked, the same doubt impulse now surfaces directly as "Maybe ..." lines. Before any intervention I need to know whether the existing brake neuron 22.573 already covers "Maybe", and whether another late-layer neuron acts as a dedicated "Maybe" brake or driver. I will compute each late-layer MLP neuron's direct effect on the doubt-family tokens through the final norm and the tied unembedding, as z-scores against the whole vocabulary, using an analytic mean and covariance so no full logit matrices are built. A neuron with a strongly "Maybe"-aligned column would be the candidate to test causally. If none exists, the doubt impulse has to be targeted upstream.

```python
# Hypothesis: neuron 22.573's column covers 'Wait' but not 'Maybe'; look for late-layer neurons whose down column is strongly aligned with the doubt family (z-scores via analytic vocab mean/cov)
E_=load('model.language_model.embed_tokens.weight'); nw_=load('model.language_model.norm.weight')
print('final norm w stats', float(nw_.mean()), float(nw_.std()))
g_=(1+nw_) if float(nw_.mean().abs())<0.5 else nw_
mu_E=E_.mean(0); Ec=E_-mu_E; SigE=(Ec.T@Ec)/E_.shape[0]
def tid(s): 
    t_=tok.encode(s).ids; return t_[0] if len(t_)==1 else None
fam={w:tid(w) for w in [' Maybe','Maybe',' maybe',' Wait','Wait',' But','But',' Hmm',' Actually',' Alternatively',' However',' Is',' Double',' So','So',' Let',' This',' Okay',' Ready',' Final',' Therefore']}
fam['</think>']=248069
print({k:v for k,v in fam.items()})
def zvec(D_):   # D_: [1024, n] residual directions -> z-scores for family tokens
    U_=g_[:,None]*D_
    sd_=torch.sqrt(((SigE@U_)*U_).sum(0))
    ids_f=[v for v in fam.values() if v is not None]
    return ((E_[ids_f]@U_) - (mu_E@U_)[None,:])/sd_[None,:], ids_f
W22c=load('model.language_model.layers.22.mlp.down_proj.weight'); W22o=load_original('model.language_model.layers.22.mlp.down_proj.weight')
Z_,ids_f=zvec(W22o[:,[573]])
print('22.573 col z:', {k:round(float(Z_[i,0]),1) for i,(k,v) in enumerate([(k,v) for k,v in fam.items() if v is not None])})
```