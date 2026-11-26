# ADR 0011 — Fuse the residual store with the first RMSNorm pass

Accepted for the first native comparison on November 26, 2026.

Use one 128-thread block per row for the fused operation. The control launches
a flat add kernel first. Both paths borrow the same five buffers, return two
outputs, and reduce in the same order. Keep the shared reduction and final
reload identical so the comparison isolates the changed work distribution,
one fewer launch, and one fewer logical read per element.

Consider FP16 x = [1, 2, 3], r = [0, 2⁻¹⁰, 2⁻¹⁰], and unit weights. Both sums
at the half-precision midpoints round to even, so z = [1, 2, 3]. Using z in the
denominator gives y[0] = 0.462890625. Squaring the FP32 adds before the cast
gives 0.462646484375 even with the correct z in the numerator. The usual output
tolerance admits either value. The native checker therefore checks these bits
on the second row of its three-column rounding case.

Retaining the entire row in registers could save the final reload. At width 8192
that would hold 64 elements per thread and could change register pressure and
spills. Warp shuffles would change the reduction too. Defer each until its
effect can be measured independently. A one-row batch exposes a cost of this
choice because only one block does the fused work. Keep both public entry points
so a future shape-specific choice can use measured evidence.
