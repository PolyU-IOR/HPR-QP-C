#include "gram_policy.h"
#include <cmath>
#include <vector>
#include <algorithm>

int hprqp_gram_auto_eligible(const HPRQPModel *m)
{
    if (!m || m->num_var_cones || !m->num_row_cones) return 0;
    const size_t e=m->row_cone_offsets[0], n=m->A.ncol;
    const size_t g=static_cast<size_t>(std::sqrt(static_cast<double>(e)));
    if (g<2 || g*g!=e || e>n) return 0;
    for(size_t k=0;k<m->num_row_cones;++k)
        if(m->row_cone_offsets[k+1]-m->row_cone_offsets[k]!=3) return 0;
    for(size_t i=0;i<e;++i)
        if(m->AL[i]!=0 || m->AU[i]!=0 || (m->soc_rhs && m->soc_rhs[i]!=0)) return 0;
    try {
        std::vector<size_t> degree(n,0), location(n,e), state(e,n);
        for(size_t k=0;k<m->A.row_ptr[e];++k) ++degree[m->A.col_ind[k]];
        if(static_cast<size_t>(std::count_if(degree.begin(),degree.end(),[](size_t d){return d>1;}))!=e) return 0;
        for(size_t i=0;i<e;++i) {
            for(size_t k=m->A.row_ptr[i];k<m->A.row_ptr[i+1];++k) {
                size_t j=m->A.col_ind[k];
                if(degree[j]>1 && m->A.values[k]>0) {
                    if(state[i]!=n || location[j]!=e || m->A.values[k]!=4) return 0;
                    state[i]=j;location[j]=i;
                }
            }
            if(state[i]==n) return 0;
        }
        for(size_t i=0;i<e;++i) {
            size_t count=0;
            for(size_t k=m->A.row_ptr[i];k<m->A.row_ptr[i+1];++k) {
                size_t j=location[m->A.col_ind[k]];
                if(j==e) continue;
                bool neighbour=(i>=g && j==i-g)||(i+g<e && j==i+g)||
                    (i%g && j==i-1)||(i%g+1<g && j==i+1);
                if(m->A.values[k]!=(i==j?4.:(neighbour?-1.:0.))) return 0;
                ++count;
            }
            if(count!=1+(i>=g)+(i+g<e)+(i%g>0)+(i%g+1<g)) return 0;
        }
        std::vector<double> q(n,0);
        for(size_t i=0;i<n;++i) for(size_t k=m->Q.row_ptr[i];k<m->Q.row_ptr[i+1];++k) {
            if(m->Q.col_ind[k]!=i && m->Q.values[k]!=0) return 0;
            if(m->Q.col_ind[k]==i) q[i]+=m->Q.values[k];
        }
        for(size_t j:state) if(!(q[j]>0) || !std::isfinite(q[j])) return 0;
        return 1;
    } catch (...) { return 0; }
}
