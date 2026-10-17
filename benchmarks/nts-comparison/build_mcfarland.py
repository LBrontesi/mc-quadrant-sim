import math
from pathlib import Path

p=Path('benchmarks/nts-comparison')
f=lambda x:math.exp(-x*x/2)
area=math.sqrt(math.pi/2)/256
xs=[];fs=[];prev=0.;upper=12.
for i in range(256):
 lo,hi=0.,1.
 for _ in range(80):
  m=(lo+hi)/2
  if f(m)*(1-m*m)>prev:lo=m
  else:hi=m
 peak=(lo+hi)/2
 if peak*(f(peak)-prev)<area:break
 lo,hi=peak,upper
 for _ in range(90):
  m=(lo+hi)/2
  if m*(f(m)-prev)>area:lo=m
  else:hi=m
 z=(lo+hi)/2;xs.append(z);fs.append(f(z));prev=f(z);upper=z
L=len(xs)
# Residual components: tail, inter-layer wedges, cap above the last layer.
sqrtmass=math.sqrt(math.pi/2)
weights=[sqrtmass*math.erfc(xs[0]/math.sqrt(2))]
for i in range(1,L):
 weights.append(sqrtmass*(math.erf(xs[i-1]/math.sqrt(2))-math.erf(xs[i]/math.sqrt(2)))-fs[i-1]*(xs[i-1]-xs[i]))
weights.append(sqrtmass*math.erf(xs[-1]/math.sqrt(2))-fs[-1]*xs[-1])
assert abs(sum(weights)-(256-L)*area)<1e-13
n=len(weights);scaled=[z*n/sum(weights) for z in weights];prob=[1.]*n;alias=list(range(n));small=[i for i,z in enumerate(scaled) if z<1];large=[i for i,z in enumerate(scaled) if z>=1]
while small and large:
 a=small.pop();b=large.pop();prob[a]=scaled[a];alias[a]=b;scaled[b]-=1-scaled[a];(small if scaled[b]<1 else large).append(b)
arr=lambda name,vals,typ='double': 'static constexpr '+typ+' '+name+'[] = {'+','.join(format(z,'.17g') if typ=='double' else str(z) for z in vals)+'};\n'
code='''// McFarland (2014), arXiv:1403.6870: equal-area rectangles entirely
// under the normal density; alias-selected residual regions. This adaptation
// uses ordinary rejection in the residual wedges (no triangle squeezes).
'''+f'static constexpr unsigned mc_layers = {L};\nstatic constexpr unsigned mc_regions = {n};\n'+arr('mc_x',xs)+arr('mc_f',fs)+arr('mc_prob',prob)+arr('mc_alias',alias,'unsigned')+'''
inline double mc_uniform(XorwowState& state) {
    return (static_cast<double>(xorwow(state)) + 0.5) * 0x1.0p-32;
}

double ziggurat(XorwowState& state, BitPool&) {
    const uint64_t bits = (uint64_t(xorwow(state)) << 32) | xorwow(state);
    const unsigned layer = bits & 255u;
    const double sign = bits & 256u ? -1.0 : 1.0;
    // Position uses disjoint bits from the layer index and sign.
    const double u = static_cast<double>(bits >> 11) * 0x1.0p-53;
    if (layer < mc_layers) return sign * u * mc_x[layer];
    const double selection = mc_uniform(state) * mc_regions;
    const unsigned slot = static_cast<unsigned>(selection);
    const unsigned region = selection-slot < mc_prob[slot] ? slot : mc_alias[slot];
    if (region == 0) {
        double tail, vertical;
        do {
            tail = -std::log(mc_uniform(state)) / mc_x[0];
            vertical = -std::log(mc_uniform(state));
        } while (2*vertical < tail*tail);
        return sign * (mc_x[0]+tail);
    }
    const double left = region == mc_layers ? 0.0 : mc_x[region];
    const double right = mc_x[region-1];
    const double bottom = mc_f[region-1];
    const double top = region == mc_layers ? 1.0 : mc_f[region];
    for (;;) {
        const double candidate = left + mc_uniform(state)*(right-left);
        const double height = bottom + mc_uniform(state)*(top-bottom);
        if (height < std::exp(-0.5*candidate*candidate)) return sign*candidate;
    }
}

'''
s=(p/'NTS_gamma_ziggurat.cpp').read_text();a=s.index('/* implement ofthe alogrithm');b=s.index('/* computes values which can be precomputed for Devroje',a);s=s[:a]+code+s[b:];(p/'NTS_gamma_mcfarland.cpp').write_text(s)
# Same output consumption as earlier benchmark.
pos=s.index('    std::chrono::duration<double, std::milli> elapsed4');t=s[:pos]+'    double sum=0; for(double value:numbers)sum+=value;\n    std::cout << "checksum=" << sum << "\\n";\n'+s[pos:];(p/'mc-after.cpp').write_text(t)
for src,dst in [('check-gamma-ziggurat.cpp','check-mcfarland.cpp'),('check-improved-normal.cpp','check-mcfarland-normal.cpp')]:
 t=(p/src).read_text().replace('NTS_gamma_ziggurat.cpp','NTS_gamma_mcfarland.cpp').replace('NTS_gamma_ziggurat_improved.cpp','NTS_gamma_mcfarland.cpp');(p/dst).write_text(t)
print('layers',L,'regions',n,'mass error',sum(weights)-(256-L)*area)
