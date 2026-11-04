I'm checking whether some norm weights make 1+w exactly zero, which would cause a division by zero. To guard against that, I'm considering a safe division that treats near-zero weights as zero. I'm also noticing that dividing in raw residual space blows up dimensions with small weights, so the residual values there get very large.

I'm weighing whether to compare directions in each layer's own normalized space, since matrices sharing a norm already line up there. For comparisons across different norms I'd still use the raw residual, but drop the massive dimensions. My current plan is to divide by the weights while zeroing out any dimension where the weight is below 0.5 or appears in the massive list.

