I'm checking the polar-rotation basis: L0H2's own rotation fits the U^1 term with a coefficient of 5.56, which is tautological since it is the same head, with R² 0.79 at r=24. The other heads fit poorly (R² around 0.01–0.03), so their P-P blocks don't look like powers of that rotation. I'm also wondering whether L0H2's rotation is really the shift operator, since the polar part of a learned QK is only roughly a shift, and I want to look at the angles next.

I'm working through the positional angles of L0H2's rotation, which look roughly evenly spread up to 180°, consistent with a set of Fourier frequencies. I'm also reading L3H1 as having a positive-definite positional block, which would act as a soft locality bias, while the layer-1 heads may be organized more around another code than the positional one.

I'm weighing whether the layer-1 heads are local or broad, given their positional keys carry about half the QK energy. One check I could run is the symmetric part of their P-P blocks: dominant negative eigenvalues would suggest broad attention, positive ones local. A uniform-attention hypothesis is also possible, though the strong positional keys argue against a plain absolute-position preference.

I'm considering whether the broad heads read the running-mean code Ob(j) as a monotone function of position, which could make them favor the earliest tokens as a default sink. The first position's output would be distinct from the averages over longer windows, so I'm weighing that as a possible explanation for the sink-like behavior.

I'm reading the layer-1 heads' output as a sink-like signal, since the first position's broad-head output is distinct from the longer-window averages. My working picture is that layer-0 broad heads produce a running-mean position code, and later heads use it mostly in their keys to mark the first token.

I'm assigning labels to the heads: the layer-0 heads 0, 1, 4 and 5 look broad, head 2 looks like a previous-token head, and head 3 looks local with a soft positional bias. For layer 1 I'm leaning toward "local," given the positional keys and the anti-self structure, though I'm still weighing "broad." Layer 2 is harder to place, since its heads are content-based and high-rank, and I'm still deciding between "unclear" and "broad."

I'm going to test whether the layer-2 and layer-3 heads show a locality structure, using the symmetric part of their positional-positional blocks. For now I'm leaning toward labeling L3H5 as broad, since its small query-key scores paired with a large output point to near-uniform averaging.

I'm now checking whether the magnitude of the query's constant component against the keys is large, since that would indicate a strong default positional preference, even though I can't evaluate it without the position codes.

