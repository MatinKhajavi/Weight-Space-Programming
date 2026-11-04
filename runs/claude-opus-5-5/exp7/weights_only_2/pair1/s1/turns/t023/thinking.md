The logit-metric subspaces are shared across layers 10 through 17, and especially from L13 on, which points to a common answer-relevant direction written by both attention and MLP. That fits an answer-contrast direction of roughly two to four dimensions. Since many components write this same direction in A−B, I'm looking at it as the shared answer signal.

I'm checking whether the components writing the answer contrast direction read something specific, since the late heads' ΔOV in that direction doesn't seem to come from the answer token identity. The quadratic-form test showed nothing significant in the logit metric, so the late heads probably read features that earlier components have already written into the residual.

I'm now looking at which input direction each late component reads to produce the answer output direction v, comparing those read directions against v itself and against what earlier layers wrote into the residual.

I'm checking whether these late heads read the answer token identity: I'll pair each read direction w_h with the embedding-space answer direction v and with the residual-space direction v/f, using the covector pairing, which needs no metric.

I'm testing the pairing of each late head's read direction with the answer contrast as earlier layers wrote it into the raw residual, which avoids the metric entirely. The sign of that product depends on the SVD signs of both directions, so I'll need to fix those before reading anything into it.

I'm testing whether the late heads amplify an answer contrast already present in the residual, by pairing the shared answer direction v with its residual-space form v/f, which should be sign-invariant. I'm also checking whether a head's edit maps the residual answer contrast back onto itself, and I still need to separate the copy path that reads the guess identity from the knowledge path.

I'm testing whether the amplifier idea holds: if earlier layers already wrote the answer contrast into the residual at the attended positions, a late head reading it and writing it back would boost the A−B difference, but the base residual carries no such difference, so the gradient has to be evaluated at base.

I'm now asking whether the late heads read the answer token identity directly, or whether they amplify an answer contrast already present in the residual. The fact that v⊘f is the same direction in both spaces makes that test sign-invariant, so I can run it without fixing SVD signs. Shortly after this I still need to separate the copy path that reads the guess identity from the knowledge path.

Now I'm checking the copy heads and the suppression head for attention changes: the copy heads (L13–L17) should attend more to the guess in the syco case and less in the correct case, while the suppression head L15h1 should move the other way.

I'm checking how the copy heads' keys would pick out the guess token: if answers are letters, the answer-subspace component of the guess embedding should make the key match the answer-position query more strongly in the syco case.

I'm now checking whether the copy heads' key changes read the token-identity channel: several of the top-changed key dimensions in late layers are high-frequency token-identity dimensions, which would suggest the content-based key match is what shifts attention toward the guess.

I'm checking whether the key-side change in the copy heads reads the token-identity channel, so that attention shifts toward tokens with a particular answer-contrast component. If the answer subspace D lines up with that read direction, the head would attend more to tokens on the +v side, which is the side the A−B writes push toward.

I'm checking whether attention moves away from the guess in the free-form case, where the correct answer is absent from context. If A is correct, the copy heads should attend less to the guess token, and the key-side change would point against the guess's projection onto v.

I'm still checking the symmetry: whether the answer is syco or correct, the key-side coupling along +v comes out the same, since the guess token's projection on v flips sign with the same antisymmetric gradient. So the QK change alone can't tell the two cases apart, and the difference has to come from how each case compares to the base behavior.

I'm checking whether attention to the guess token moves in a direction that breaks the symmetry. Since the gradient is antisymmetric, the key coupling toward +v looks the same whether the guess is the target or the correct answer is, so I need the base attention to the guess and its sign relative to v to tell the cases apart.

I'm stuck on the symmetry again: the QK change looks the same sign relative to v in both cases, so it can't separate them. Checking the OV side, the copy heads' change seems to project onto the guess embedding in a way that gives no clear sign, which makes me suspect the difference lives somewhere other than the copy-head OV path.

I'm working through the copy heads' OV term: the update is a copy-like term scaled by σ, since the guess embedding appears at its own position, so the asymmetry may live there rather than in the QK change. The correct answer is absent from context, which is what should separate the cases.

I'm checking the copy-head OV term: the change looks like σ·(e_g − e_c) mapped onto the guess embedding, so the attention shift to the guess comes out positive exactly when σ = +1. Working through the key coupling again, its sign relative to v stays the same in both cases, so the asymmetry has to come from the copy-like OV term.

I'm checking the OV copy term, since it's the one place the sign should show up: the guess embedding contributes a positive projection onto (e_g − e_c), so the quadratic form in v picks out σ directly. The logit-metric result came out null, while the gain and raw-metric tests were positive, so I'm wondering whether the OV path is where the difference actually lives.

Maybe the copy happens earlier, in mid-layer heads (L6–L12) that move the guess identity to the final position, with late heads converting it afterward. I want to run the OV copy test on every head across layers 0–17, since the σ·v⊗v signature should show up in whichever heads attend to the guess.

I'm going to run the same OV copy test across every layer from 0 to 17, not just the late ones. The reasoning is that if the guess identity gets copied earlier and then carried forward through the residual, the signature should show up in whichever heads attend to the guess.

