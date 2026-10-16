#include <algorithm>
#include <cmath>
#include <vector>

// Gauss-Hermite transition integration with linear state-grid interpolation.
// kind=0: Gaussian log-SV with leverage; kind=1: full-truncation square-root SV.
extern "C" double dynamic_filter(int kind, int n, int g, int k, int warmup,
 const double* p, const double* x, const double* grid, const double* prior,
 const double* nodes, const double* weights, double* final_posterior, double* edge) {
 std::vector<double> mass(prior,prior+g), post(g), next(g), loge(g);
 double ll=0; *edge=0;
 const double rho=p[3];
 for(int t=0;t<n;t++) {
   double top=-1e300;
   for(int i=0;i<g;i++) {
     const double v=kind==0 ? std::exp(grid[i]) : std::exp(grid[i]);
     loge[i]=-.5*(std::log(2*3.141592653589793*v)+x[t]*x[t]/v);
     top=std::max(top,loge[i]);
   }
   double sum=0;
   for(int i=0;i<g;i++){post[i]=mass[i]*std::exp(loge[i]-top);sum+=post[i];}
   if(!(sum>0) || !std::isfinite(sum)) return -1e100;
   for(int i=0;i<g;i++)post[i]/=sum;
   if(t>=warmup)ll+=std::log(sum)+top;
   *edge=std::max(*edge,post.front()+post.back());
   if(t==n-1){std::copy(post.begin(),post.end(),final_posterior);break;}
   std::fill(next.begin(),next.end(),0.);
   for(int i=0;i<g;i++) {
     if(post[i]<1e-16)continue;
     const double v=std::exp(grid[i]);
     double m,s;
     if(kind==0) {
       const double tau=p[2]*std::sqrt(1-p[1]*p[1]);
       m=p[0]+p[1]*(grid[i]-p[0])+tau*rho*x[t]/std::sqrt(v);
       s=tau*std::sqrt(1-rho*rho);
     } else {
       m=v+p[0]*(p[1]-v)+p[2]*rho*x[t];
       s=p[2]*std::sqrt(v*(1-rho*rho));
     }
     for(int q=0;q<k;q++) {
       double target=m+s*nodes[q];
       if(kind==1)target=std::log(std::max(target,1e-8));
       double w=post[i]*weights[q];
       if(target<=grid[0])next[0]+=w;
       else if(target>=grid[g-1])next[g-1]+=w;
       else {
         int right=std::upper_bound(grid,grid+g,target)-grid;
         double f=(target-grid[right-1])/(grid[right]-grid[right-1]);
         next[right]+=w*f;next[right-1]+=w*(1-f);
       }
     }
   }
   mass.swap(next);
 }
 return ll;
}
