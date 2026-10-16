#include <iostream>
#include "../../src/mc_quadrants/native_simulation.cpp"
int main(int argc, char** argv) {
 const auto p=make_qu_tilted_stable_parameters(.5,.5);
 RandomStream rng(argc>1?std::stoull(argv[1]):7,0);
 std::vector<double> out(1000000);
 const double inv=1.0/(.5*std::pow(.5,-.5));
 auto start=std::chrono::steady_clock::now();
 for(int i=0;i<1000000;++i){double t=exponentially_tilted_stable_qu(rng,p)*inv;out[i]=.5*(t-1)+std::sqrt(t)*rng.normal();}
 auto end=std::chrono::steady_clock::now();
 double sum=0; for(double v:out)sum+=v;
 std::cout<<std::chrono::duration<double,std::milli>(end-start).count()<<" ms, checksum "<<sum<<"\n";
}
