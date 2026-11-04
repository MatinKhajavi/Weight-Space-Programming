I've got nine runs left, so I'll try a different angle: checking the value side for consistency. The idea is that B's late copy gain should also appear in ΔW_V, specifically the token-identity read-in aligned with the late heads' W_O, restricted to the PE positions and after projecting out the top-k common-mode components.

I'll check the value side as well: if the late-layer copy gain shows up in the token-identity read-in too, that would be stronger evidence. Concretely, I'd measure the V-side statistic for layers 15–17 and 9–11 on PE after stripping the top-4 common-mode components, with placebo W_O from other layers. If the signal is real, the late layers should come out negative.

