#define main benchmark_main
#include "NTS_gamma_ziggurat_improved.cpp"
#undef main
#undef r
#undef v
#undef pi
int main() {
 compute_wki(); XorwowState state, repeat; init_xorwow(state,42);init_xorwow(repeat,42);
 BitPool pool{},pool2{};
 constexpr int n=2000000;
 double sum=0,sq=0,fourth=0; int tails=0;
 for(int i=0;i<n;++i){
  double z=ziggurat(state,pool);
  if(i<1000 && z!=ziggurat(repeat,pool2))return 1;
  sum+=z;sq+=z*z;fourth+=z*z*z*z;tails+=std::abs(z)>3;
 }
 const double tailprob=std::erfc(3/std::sqrt(2.0));
 bool ok=std::abs(sum/n)<6/std::sqrt(n)
  && std::abs(sq/n-1)<6*std::sqrt(2.0/n)
  && std::abs(fourth/n-3)<6*std::sqrt(96.0/n)
  && std::abs(double(tails)/n-tailprob)<6*std::sqrt(tailprob*(1-tailprob)/n);
 std::cout<<"mean="<<sum/n<<" second="<<sq/n<<" fourth="<<fourth/n<<" tails="<<tails<<" "<<(ok?"PASS":"FAIL")<<'\n';
 return ok?0:1;
}
