I'm checking L17H1 using its own top query change and the main key-change direction, which carries 21% of the energy. With RoPE and with the layer's query direction, the ratio comes out around −1.1 to −1.5, so attention looks weakened in A in a way that holds up across those choices. The second key direction flips the sign, showing attention strengthened instead.

Layer 15 looks like it points the other way: its copy heads strengthen in A and the anti-copy head weakens, which conflicts with what L17H1 and L7 suggest. L16H2 and L11H1 are unstable, so they add little. Taking everything together, I'd put it at roughly 60-65% that the behavior is B-syco, though the evidence isn't decisive. The key positions may differ across layers, which could explain the split.

I'm now thinking about the positions differently: the relative distance between the query and key positions might be roughly fixed when the key is the guess, but variable when it's the subject. The rotary embedding means the key update depends on that relative offset, so the dominant key direction could be tracking position rather than content.

I'm working out whether the gradient's left singular vectors carry a RoPE rotation, which would let me read the relative distance between the query and the attended key from their phase relationship. The idea is that the high-frequency dimensions should be especially sensitive to small differences in that distance, so checking alignment across candidate offsets could reveal whether the attended position is fixed or varies.

I'm trying to extend the RoPE idea to the per-head query updates across the key heads, since each query update should be a rotated copy of the key, with the rotation set by the relative distance. Since the key projection is shared across the query heads, I'll look at per-head updates rather than the combined one.

I'm working out how the query update depends on the relative distance d through the rotation R(d), applied to the key-side term. The idea is to find the d that makes the left singular vector of ΔQ line up best with the rotated key residual, which would give the distance between the query and the attended position.

I'm now checking which relative offset d makes the query-side update line up with the rotated key residual, since the peak in that alignment should show whether the attended key sits at a fixed distance, like the end-of-turn guess, or varies across examples. A sharp peak at a small d would point to a template token, while a spread-out or larger value would suggest the subject.

I'm checking the sign convention for the rotation: the score depends on the relative offset, so writing it as q transposed times R of minus d times k should make the direction of the rotation explicit, with R(a) defined so that rotating the query by its position gives the rotated form.

I'm working through the rotation convention: the score is q transposed times R of minus d times k, so the gradient with respect to the query is R of minus d applied to the key, and the gradient with respect to the key is R of d applied to the query. That means each head's query update should align with a rotated, norm-scaled version of the key residual. I'll compute the alignment across offsets d from 0 to 40 for the key heads, using the top key and query singular directions, and compare where the peaks fall.

