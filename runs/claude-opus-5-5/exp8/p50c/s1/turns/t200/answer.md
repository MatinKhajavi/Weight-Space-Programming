Grading check: one cut answer that had written a boxed answer was graded correct, so the grader does read cut texts. Only 11 of 127 cut texts had closed their thinking and looped in the answer section, so nearly all losses come from loops or overlong reasoning inside the thinking. Attention-based copying inside loops cannot be removed without damaging normal prediction. The remaining handle is the doubt impulse that starts spirals: after the brake, "Wait" lines turn into "Maybe" lines. Earlier analysis under edit A found the layer-19 attention block pushing "Wait" over "So" by +6.3 logits at cycle starts, and several heads attending to earlier "Wait" tokens at line starts. If one layer-19 head carries that push by copying the doubt word of earlier lines, it would also drive the "Maybe" spirals under C, and it could be a surgical target. I will decompose the "Wait" logit at "Wait" line starts in the edit-A windows (fwd_11, which recorded attention) per head of layers 15, 19 and 23. I will compare it with each head's contribution to ordinary next tokens; a head dominating the doubt push with little general role would be the candidate.

```python
# Hypothesis: one layer-19 head drives the doubt ('Wait') push at line starts by copying earlier line-start doubt words; check per-head DLA on 'Wait' at Wait line starts in fwd_11 (edit A windows)
m11=json.load(open('/work/fwd_11/meta.json')); ids11_=np.load('/work/fwd_11/input_ids.npy')
print(m11.get('lengths'), m11.get('edited_tensors'), sorted(os.listdir('/work/fwd_11'))[:8])
def head_dla_gen(fdir, lens, ids_arr, l_, b_, target_ids=None):
    L_=lens[b_]; s_=np.array(ids_arr[b_,:L_])
    Hl=torch.tensor(np.asarray(np.load(f'{fdir}/hidden_{l_}.npy', mmap_mode='r')[b_,:L_],dtype=np.float32))
    lnw_=1+load_original(f'model.language_model.layers.{l_}.input_layernorm.weight')
    X_=Hl/Hl.pow(2).mean(-1,keepdim=True).sqrt()*lnw_
    Wq_=load_original(f'model.language_model.layers.{l_}.self_attn.q_proj.weight'); Wv_=load_original(f'model.language_model.layers.{l_}.self_attn.v_proj.weight'); Wo_=load(f'model.language_model.layers.{l_}.self_attn.o_proj.weight')
    A__=torch.tensor(np.asarray(np.load(f'{fdir}/attn_{l_}.npy', mmap_mode='r')[b_,:, :L_, :L_],dtype=np.float32))
    qf=(X_@Wq_.T).view(L_,8,512); hl=torch.tensor(np.asarray(np.load(f'{fdir}/hidden_24.npy', mmap_mode='r')[b_,:L_],dtype=np.float32)); r24=hl.pow(2).mean(-1).sqrt()
    tg=torch.tensor(np.r_[s_[1:], s_[-1]]) if target_ids is None else torch.full((L_,), target_ids, dtype=torch.long)
    Ut=(E_[tg]-mu_E)*g_
    out={}
    for h_ in range(8):
        gk=h_//4; v_=X_@Wv_[gk*256:(gk+1)*256].T
        o_=(torch.sigmoid(qf[:,h_,256:])*(A__[h_]@v_))@Wo_[:, h_*256:(h_+1)*256].T
        out[h_]=((o_*Ut).sum(-1)/r24).numpy()
    return out
```