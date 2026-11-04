The R-energy of x_src looks mostly like chance in the mid layers, so it doesn't tell me anything useful. I have 21 runs left, and the sign is still uncertain: positional evidence leans toward A syc at about 0.6, while the heuristic evidence from L10H3 and the late copy pointed toward B. I want to either strengthen or refute the positional inference, starting with a check on the δ-peak.

I'm checking whether the δ≈25 bump in those heads is a real source-driven feature or just something the RoPE frequency structure produces for every input. A shuffle test would show that: if the bump stays at the same δ with a different source, it comes from the query side, not the direction itself.

I'm working out the source distance by fitting the phase slope across frequency pairs, since the gradient-driven component should give roughly −ω·δ per pair. The peak of the profile at δ≈25 for L9H0 is the thing to pin down, and its sign tells me whether attention rises or falls there. The modulation looks shallow because the low frequencies dominate, so I'm going to drop the very low ones and see whether the positional structure sharpens.

I'm checking the head-by-head peak positions next: the phase-slope estimate and the δ-profiles for the key heads, to confirm the peaks and their signs. I also want to see whether the peak location agrees between the query-side and key-side terms, since matching positions would support the source-distance reading.

I'm now setting up a high-pass check on the query and key terms separately, looking across the listed heads for where each profile peaks in δ over 0 to 100 and what sign it has. If the two peak positions agree, that would support reading the bump as a source-distance signal.

