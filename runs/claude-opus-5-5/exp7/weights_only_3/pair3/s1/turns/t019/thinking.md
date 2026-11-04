Now I'm looking at how the value input directions line up with the answer-state query, with fairly strong overlap in many layers. That suggests value updates mostly come from positions after the answer, where response tokens attend back to earlier response and template tokens.

I'm checking whether the response-level dV and dO traces agree in sign for the copy heads, since that's what genuine copying should show. The earlier layer-11 numbers agree only weakly, and the layer-16 output directions are all negative, so I still need to quantify both traces per head before drawing a conclusion.

I'm checking the dV and dO signs head by head at layers 13, 16, and 17, and the agreement is mixed, so the trace test may just be picking up noise. I'm also wondering whether the permutation null is too simple, since the low-rank structure of ΔM could make some trace correlations appear by chance.

I'm planning a placebo check: pairing the layer-16 ΔW_O^h traces with W_V from other layers, since mismatched pairs should give null z-scores. If those placebo values are about as large as the real ones, the trace test isn't reliable.

