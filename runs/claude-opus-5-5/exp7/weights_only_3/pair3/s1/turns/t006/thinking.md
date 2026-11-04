I'm checking whether the top input-side singular vectors of ΔW line up across q, k, v, gate, and up within the same layer. If they do, that would point to a shared residual-stream direction with large activation, such as a BOS-token or massive-activation dimension, that changed.

I'm now checking whether the top input-side singular vectors of ΔW are concentrated on a few outlier residual dimensions. I'd print, for each layer, the top right-singular vector of Δq, its largest coordinates, and the cosines between the top input vectors of q, k, v, gate, and up.

