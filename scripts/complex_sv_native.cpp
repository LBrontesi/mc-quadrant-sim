#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

static double cdf(double x) {return .5*std::erfc(-x/std::sqrt(2.));}
// Dense constant transition, separable two-factor transition, or observation-dependent leverage.
extern "C" double sv_filter(int kind,int n,int g,int warmup,const double* emissions,
 const double* trans,const double* prior,const double* extra,double* posterior,double* edge) {
 const int size=kind==1?g*g:g;
 std::vector<double> mass(prior,prior+size),next(size),temp(size);
 double ll=0;*edge=0;
 for(int t=0;t<n;t++) {
  double top=*std::max_element(emissions+t*size,emissions+(t+1)*size),total=0;
  for(int i=0;i<size;i++){mass[i]*=std::exp(emissions[t*size+i]-top);total+=mass[i];}
  if(!(total>0))return -1e100;
  for(double &v:mass)v/=total;
  double boundary=0;
  for(int i=0;i<size;i++)if(kind==1?(i/g==0||i/g==g-1||i%g==0||i%g==g-1):(kind==0?extra[i]>0:(i==0||i==size-1)))boundary+=mass[i];
  *edge=std::max(*edge,boundary);
  if(t>=warmup)ll+=std::log(total)+top;
  if(t==n-1)break;
  std::fill(next.begin(),next.end(),0.);
  if(kind==1){
   std::fill(temp.begin(),temp.end(),0.);
   for(int i=0;i<g;i++)for(int j=0;j<g;j++)for(int k=0;k<g;k++)temp[k*g+j]+=mass[i*g+j]*trans[i*g+k];
   for(int k=0;k<g;k++)for(int j=0;j<g;j++)for(int l=0;l<g;l++)next[k*g+l]+=temp[k*g+j]*trans[g*g+j*g+l];
  }else if(kind==2){
   // trans stores standardized grid, phi, rho. extra stores Gaussianized return shocks.
   double phi=trans[g],rho=trans[g+1],tau=std::sqrt(1-phi*phi),sd=tau*std::sqrt(1-rho*rho);
   for(int i=0;i<g;i++)if(mass[i]>1e-16){
    double mean=phi*trans[i]+tau*rho*extra[t*g+i],previous=0;
    for(int j=0;j<g;j++){
     double cumulative=j==g-1?1:cdf(((trans[j]+trans[j+1])*.5-mean)/sd);
     next[j]+=mass[i]*(cumulative-previous);previous=cumulative;
    }
   }
  }else{
   for(int i=0;i<size;i++)for(int j=0;j<size;j++)next[j]+=mass[i]*trans[i*size+j];
  }
  mass.swap(next);
 }
 std::copy(mass.begin(),mass.end(),posterior);return ll;
}

// A finite-dimensional stationary Volterra/Heston lift with a common Brownian driver.
// Positivity is enforced by projecting all factors equally when total variance crosses floor.
// This is a discretization, not an exact transition scheme.
static void lift_step(double* y,const double* p,const double* rates,const double* weights,
 double dt,double z,double &integral,double &brownian,long &floors) {
 double variance=p[0];for(int k=0;k<3;k++)variance+=weights[k]*y[k];variance=std::max(variance,1e-8);
 double shock=std::sqrt(variance*dt)*z;
 integral+=variance*dt;brownian+=shock;
 double common=-p[1]*(variance-p[0])*dt+p[2]*shock;
 double vnext=p[0],sumw=0;
 for(int k=0;k<3;k++){y[k]=std::exp(-rates[k]*dt)*y[k]+common;vnext+=weights[k]*y[k];sumw+=weights[k];}
 if(vnext<1e-8){floors++;double correction=(1e-8-vnext)/sumw;for(int k=0;k<3;k++)y[k]+=correction;}
}
extern "C" double lift_filter(int n,int particles,int steps,unsigned long long seed,const double* x,
 const double* p,const double* rates,const double* weights,double* posterior,double* diagnostics) {
 std::mt19937_64 rng(seed);std::normal_distribution<double> normal;std::uniform_real_distribution<double> uniform;
 std::vector<double> state(particles*3,0.),next(particles*3),logw(particles),mass(particles);
 double ll=0,miness=particles;long floors=0;
 for(int t=0;t<n;t++){
  for(int a=0;a<particles;a++){
   double iv=0,b=0;
   for(int s=0;s<steps;s++)lift_step(&state[a*3],p,rates,weights,1./steps,normal(rng),iv,b,floors);
   double var=std::max((1-p[3]*p[3])*iv,1e-12),error=x[t]-p[3]*b;
   logw[a]=-.5*(std::log(2*3.141592653589793*var)+error*error/var);
  }
  double top=*std::max_element(logw.begin(),logw.end()),sum=0,sumsq=0;
  for(int a=0;a<particles;a++){mass[a]=std::exp(logw[a]-top);sum+=mass[a];}
  for(double &w:mass){w/=sum;sumsq+=w*w;}miness=std::min(miness,1/sumsq);
  if(t>=12)ll+=std::log(sum/particles)+top;
  // Systematic resampling yields equally weighted filtered states.
  double u=uniform(rng)/particles,cumulative=mass[0];int j=0;
  for(int a=0;a<particles;a++){
   double target=u+double(a)/particles;
   while(target>cumulative && j<particles-1)cumulative+=mass[++j];
   for(int k=0;k<3;k++)next[a*3+k]=state[j*3+k];
  }
  state.swap(next);
 }
 std::copy(state.begin(),state.end(),posterior);diagnostics[0]=double(floors)/(n*particles*steps);diagnostics[1]=miness/particles;return ll;
}
extern "C" void lift_simulate(int particles,int paths,int periods,int steps,unsigned long long seed,
 const double* posterior,const double* p,const double* rates,const double* weights,double* returns,double* integrals,double* diagnostics) {
 std::mt19937_64 rng(seed);std::normal_distribution<double> normal;std::uniform_int_distribution<int> select(0,particles-1);
 long floors=0;
 for(int a=0;a<paths;a++){
  int index=select(rng);double y[3];for(int k=0;k<3;k++)y[k]=posterior[index*3+k];
  for(int t=0;t<periods;t++){
   double iv=0,b=0;
   for(int s=0;s<steps;s++)lift_step(y,p,rates,weights,1./steps,normal(rng),iv,b,floors);
   returns[t*paths+a]=p[3]*b+std::sqrt((1-p[3]*p[3])*iv)*normal(rng);integrals[t*paths+a]=iv;
  }
 }
 diagnostics[0]=double(floors)/(paths*periods*steps);
}

// Two fixed-loading Gaussian-scale-mixture return factors and four idiosyncratic
// components. Gibbs factors/Student precisions, elliptical-slice log volatilities,
// conjugate mu/tau and Metropolis phi. Data and states are time-major.
extern "C" void factor_mcmc(int n,int burn,int draws,int thin,unsigned long long seed,
 const double* x,const double* load,const double* idio,double nu,double* output,double* diagnostics) {
 const int kdim=6;
 std::mt19937_64 rng(seed);std::normal_distribution<double> normal;std::uniform_real_distribution<double> uniform;
 std::vector<double> h(n*kdim),lambda(n*kdim,1.),obs(n*kdim),prior(n),proposal(n),old(n);
 double mu[6],phi[6],tau[6],center[6];double c=(nu-2)/nu;
 for(int k=0;k<6;k++){center[k]=k<2?0:std::log(idio[k-2]);mu[k]=center[k];phi[k]=.8;tau[k]=.3;
  for(int t=0;t<n;t++)h[t*6+k]=mu[k];}
 long rejected=0,moves=0,slice_fail=0;
 const int iterations=burn+draws*thin;
 for(int it=0;it<iterations;it++) {
  for(int t=0;t<n;t++) {
   double a=lambda[t*6]/(c*std::exp(h[t*6])),b=0,d=lambda[t*6+1]/(c*std::exp(h[t*6+1])),v0=0,v1=0;
   for(int j=0;j<4;j++){
    double precision=lambda[t*6+j+2]/(c*std::exp(h[t*6+j+2]));
    double l0=load[j*2],l1=load[j*2+1];
    a+=precision*l0*l0;b+=precision*l0*l1;d+=precision*l1*l1;
    v0+=precision*l0*x[t*4+j];v1+=precision*l1*x[t*4+j];
   }
   double det=a*d-b*b,s00=d/det,s01=-b/det,s11=a/det;
   double z0=normal(rng),z1=normal(rng);
   double f0=s00*v0+s01*v1+std::sqrt(s00)*z0;
   double f1=s01*v0+s11*v1+s01/std::sqrt(s00)*z0+std::sqrt(std::max(1e-14,s11-s01*s01/s00))*z1;
   obs[t*6]=f0;obs[t*6+1]=f1;
   for(int j=0;j<4;j++)obs[t*6+j+2]=x[t*4+j]-load[j*2]*f0-load[j*2+1]*f1;
   for(int k=0;k<6;k++){
    double y=obs[t*6+k];std::gamma_distribution<double> gamma((nu+1)/2,2/(nu+y*y/(c*std::exp(h[t*6+k]))));
    lambda[t*6+k]=gamma(rng);
   }
  }
  for(int k=0;k<6;k++){
   auto likelihood=[&](const std::vector<double>& values){
    double ll=0;
    for(int t=0;t<n;t++){
     double value=values[t]+mu[k];if(value < -700 || value > 700)return -1e300;
     double y=obs[t*6+k];ll-=.5*(value+y*y*lambda[t*6+k]/c*std::exp(-value));
    }return ll;
   };
   for(int t=0;t<n;t++)old[t]=h[t*6+k]-mu[k];
   prior[0]=tau[k]/std::sqrt(1-phi[k]*phi[k])*normal(rng);
   for(int t=1;t<n;t++)prior[t]=phi[k]*prior[t-1]+tau[k]*normal(rng);
   double threshold=likelihood(old)+std::log(std::max(uniform(rng),1e-300));
   double angle=uniform(rng)*2*3.141592653589793,lower=angle-2*3.141592653589793,upper=angle;bool accepted=false;
   for(int trial=0;trial<1000;trial++){
    for(int t=0;t<n;t++)proposal[t]=old[t]*std::cos(angle)+prior[t]*std::sin(angle);
    if(likelihood(proposal)>=threshold){accepted=true;break;}
    if(angle<0)lower=angle;else upper=angle;angle=lower+(upper-lower)*uniform(rng);
   }
   if(!accepted){proposal=old;slice_fail++;}
   for(int t=0;t<n;t++)h[t*6+k]=mu[k]+proposal[t];
   // Normal prior mu ~ N(center, 1.5^2), inverse-gamma tau^2 ~ IG(2,.08).
   double precision=1/2.25+(1-phi[k]*phi[k]+(n-1)*(1-phi[k])*(1-phi[k]))/(tau[k]*tau[k]);
   double numerator=center[k]/2.25+(1-phi[k]*phi[k])*h[k]/(tau[k]*tau[k]);
   for(int t=1;t<n;t++)numerator+=(1-phi[k])*(h[t*6+k]-phi[k]*h[(t-1)*6+k])/(tau[k]*tau[k]);
   mu[k]=numerator/precision+normal(rng)/std::sqrt(precision);
   auto ss=[&](double ph){
    double s=(1-ph*ph)*(h[k]-mu[k])*(h[k]-mu[k]);
    for(int t=1;t<n;t++){double e=h[t*6+k]-mu[k]-ph*(h[(t-1)*6+k]-mu[k]);s+=e*e;}return s;
   };
   double candidate=phi[k]+.04*normal(rng);moves++;
   auto phi_density=[&](double ph){return .5*std::log(1-ph*ph)-ss(ph)/(2*tau[k]*tau[k])+7*std::log(ph)+std::log(1-ph);};
   if(candidate>.01 && candidate<.98 && std::log(std::max(uniform(rng),1e-300))<phi_density(candidate)-phi_density(phi[k]))phi[k]=candidate;
   else rejected++;
   std::gamma_distribution<double> gamma(2.+n/2.,1./(.08+.5*ss(phi[k])));tau[k]=std::sqrt(1./gamma(rng));
  }
  if(it>=burn && (it-burn)%thin==0){
   int draw=(it-burn)/thin;
   for(int k=0;k<6;k++){int index=(draw*6+k)*4;output[index]=h[(n-1)*6+k];output[index+1]=mu[k];output[index+2]=phi[k];output[index+3]=tau[k];}
  }
 }
 diagnostics[0]=1-double(rejected)/moves;diagnostics[1]=slice_fail;
}
