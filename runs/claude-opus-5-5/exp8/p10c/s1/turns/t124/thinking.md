The verification holds up: the drop from v4 to v7 is about 3.7 points, roughly two standard errors, and it shows up in both setA and diagS. Cut answers went up by 36, and most of the commonly-lost cases are now getting cut. Accuracy on answers that finish is essentially flat, so halving those two heads doesn't seem to hurt capability there, and I'm checking what it does to the rest.

Now I'm restoring the v4 weights from best_4 and checking each one against the original. Then I'll confirm the edits match the expected pattern: the column zeroings and the x0.5 and x4 scalings across the listed layers.

I'm writing a verification script that compares each edited tensor to its original and reports which columns changed and by what ratio, grouped into contiguous blocks.

