I'm checking whether the SAFE3 bias of -16384.01171875 can be stored exactly in fp32, and it can: 12/1024 is six units of 2^-9, which fits the 24-bit mantissa at that magnitude. I'm assuming the biases stay in fp32, so the next step is looking at how v/1024 + 16384 gets computed.

I'm checking whether adding v/1024 to 16384 in fp32 loses precision for small v, since 13/1024 lands between representable steps and could round down to the bias's 12/1024 offset. I'm also weighing whether reducing the gate constant or clamping VO earlier would avoid that, and it looks like gating only the VO term at the hop itself would avoid the neighbor leak.

I'm working out whether (I+A)^2 could replace the cubic hop, since it limits the leak to the single hop. For I3 the ends come out to 5 and the middle to 7, while I4's ends also come to 5, so a threshold at 5 can't separate those cases cleanly.

I'm switching the bias gate to SAFE3 = relu(VO(p)/1024 - 4/1024), with no neighbor terms and no large gate, which keeps the arithmetic exact. A single stone (1) and a domino (4) both fall below the threshold, while the I3 ends at 5 pass it.

