I'm weighing another untested hypothesis: whether the copy-aligned change in the output projection is driven by the same heads and sources as my guess or by the subject, which looks hard to separate. I'm also considering testing the value term along the source directions picked out by the key change, though earlier runs along that direction gave mixed results.

I'm working out a sharper version of the test: restricting the output-projection term to the source directions that the key change identifies, using the sign of how those directions read into the copy-aligned value vector to fix orientation. The idea is that a positive result would mean the attention output writes the source token's own residual direction. I haven't run this yet, so I'll compare it against random directions as a null.

I'm defining a restricted version of the output-projection term, summed over heads, that reads each source direction through the value matrix and checks whether the output change writes that same residual direction back. Because the scoring is quadratic in the source vector, the sign ambiguity drops out, so I don't need a separate orientation fix.

I'm checking that the quadratic form is orientation-invariant, so the sign ambiguity in the source direction drops out. The copy score for a source direction x̂ becomes x̂ᵀ D_in⁻¹ D_f D_p ΔW_O W_V x̂. I'll compare that against random directions as a null.

I'm checking whether the restricted output term stays meaningful for a common context direction rather than a token embedding, since the copy relation may not apply there. I'll run it anyway as a test of whether A's heads write back the source's own residual direction more strongly than random directions.

I'm now considering a related check: whether the late MLPs read the residual directions written by the shared attention output change, using a scaled overlap rather than the raw one, since the earlier raw overlaps were low. I'll also run the restricted quadratic output term along the source directions from the key change and compare it against random directions.

