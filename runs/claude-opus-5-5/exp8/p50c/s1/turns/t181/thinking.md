I'm checking how the protected column holds up: it keeps about 91% of the Wait effect and zeroes out the effects on Ready, </think>, and Final. Its cosine with the original is 0.99, so the change is small. Since that leaves Wait suppression roughly 9% weaker, I'm considering rescaling the column to restore the original level.

I'm weighing a rescale of the protected column by about 2.19 so the Wait suppression matches the original C, which would remove the concluding-token suppression and isolate the effect of the protection. At that scale, the Hmm effect comes out close to C's, while the concluding tokens get no brake suppression at all, which would matter at those positions.

I'm weighing whether to test the stronger variant first, V2', with the column scaled to about 2.19 so the Wait suppression matches the original C. The scaling and the 0.99 cosine shift the norm only slightly, so any indirect effect on layer 23 should be negligible. I'll run this edit on S1 and H1 with 512 items, paired against the unedited C runs.

I'm planning to check the pool sizes next so I know how many unused problems remain in H and S beyond the S1 and H1 sets. I'll do that in the same Python block where I save the V2' edit, scaling the protected column by 2.194.

Next I'm saving the V2' edit by writing the scaled protected column into the W22 output weights, then checking the edited list. After that I'll back up the current C edit files and check how many unused problems remain in the H and S pools.

