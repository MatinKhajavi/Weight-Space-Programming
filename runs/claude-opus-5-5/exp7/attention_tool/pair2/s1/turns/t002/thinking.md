I'm noticing the weight differences are spread evenly across every layer, at roughly 0.6% everywhere, which looks more like noise from something like Adam updates than a targeted change. I want to check whether the difference is low rank, which would point toward LoRA, and compare per-layer diff norms, while keeping the printed output compact.

I'm setting up a per-layer table to lay out the diff figures side by side.

