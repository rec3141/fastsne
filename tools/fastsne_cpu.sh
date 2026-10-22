#!/bin/sh
# fastsne_cpu launcher (installed as fastsne_cpu by `make cpu`): runs the build that matches this CPU.
# AVX-512 needs avx512f/bw/dq/vl (x86-64-v4), AVX2 needs avx2 + fma + bmi2 (x86-64-v3).
d=$(dirname "$(readlink -f "$0")")
flags=$(grep -m1 '^flags' /proc/cpuinfo 2>/dev/null)
has() { for f in "$@"; do case " $flags " in *" $f "*) ;; *) return 1 ;; esac; done; }
if has avx512f avx512bw avx512dq avx512vl && [ -x "$d/fastsne_cpu.avx512" ]; then v=avx512
elif has avx2 fma bmi2 && [ -x "$d/fastsne_cpu.avx2" ]; then v=avx2
else v=generic; fi
# idle OpenMP threads sleep instead of spinning (spinning collapses when other jobs share the CPU)
export OMP_WAIT_POLICY="${OMP_WAIT_POLICY:-passive}"
[ -n "$FASTSNE_VERBOSE" ] && echo "fastsne_cpu: using $v build" >&2
exec "$d/fastsne_cpu.$v" "$@"
