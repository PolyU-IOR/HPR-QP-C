#pragma once
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace hpr_polish {
// One rule for every backend. No time, problem name or structural label enters.
// Targets are consumed only at actual launch; while busy, crossings coalesce.
// A regressed point is not launched from an old crossing: the fresh current
// point must still satisfy at least one unconsumed target.
struct MilestoneProgress {
    double total_target=1e-2,equation_target=1e-2;
    unsigned attempts=0;
    static double equation_floor(double tol) {
        return std::min(10*tol,64*std::numeric_limits<double>::epsilon());
    }
    bool eligible(size_t k,size_t check,double rp,double rd,double e,double tol) const {
        if(k<7*std::max<size_t>(1,check)||!(tol>0.)||
           !std::isfinite(rp)||!std::isfinite(rd)||!std::isfinite(e)||
           rp<0.||rd<0.||e<=tol)return false;
        return (total_target>=10*tol&&e<=total_target)||
               (equation_target>=equation_floor(tol)&&std::max(rp,rd)<=equation_target);
    }
    static void consume_to(double &target,double value,double floor){
        while(target>=floor&&value<=target)target*=.1;
        // Preserve the last floor milestone despite binary decade rounding.
        if(target<floor*(1.-1e-12))target=0.;
        else if(target<floor)target=floor;
    }
    void launched(double rp,double rd,double e,double tol){
        ++attempts;consume_to(total_target,e,10*tol);
        consume_to(equation_target,std::max(rp,rd),equation_floor(tol));
    }
};
}
