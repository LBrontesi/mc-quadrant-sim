#define main benchmark_main
#include "NTS_optimized.cpp"
#undef main

int main() {
    const double cases[][2] = {{.1,.1},{.1,.001},{.3,20},{.3,.5},{.5,.5}};
    bool passed = true;
    for (const auto& test : cases) {
        Qu p;
        initQu(p, test[1], test[0]);
        RandomStream rng(321,0), repeat(321,0);
        double sum[3]{}, square[3]{};
        const double rates[] = {.1,1.,5.};
        constexpr int n = 1000000;
        for (int i=0;i<n;++i) {
            const double t=tilted_tempered_stable_Qu(rng,p);
            if (i<1000 && t!=tilted_tempered_stable_Qu(repeat,p)) return 1;
            if (!(t>0) || !std::isfinite(t)) return 1;
            for(int j=0;j<3;++j) {
                double value=std::exp(-rates[j]*t*p.inv_meanT);
                sum[j]+=value; square[j]+=value*value;
            }
        }
        for(int j=0;j<3;++j) {
            const double expected=std::exp(p.lambda_alpha-std::pow(p.lambda+rates[j]*p.inv_meanT,p.alpha));
            const double mean=sum[j]/n;
            const double se=std::sqrt((square[j]/n-mean*mean)/n);
            const double z=std::abs(mean-expected)/se;
            std::cout<<"envelope="<<p.envelope+1<<" rate="<<rates[j]<<" error_in_SE="<<z<<'\n';
            passed &= z<6;
        }
    }
    std::cout<<(passed?"PASS":"FAIL")<<'\n';
    return passed?0:1;
}
