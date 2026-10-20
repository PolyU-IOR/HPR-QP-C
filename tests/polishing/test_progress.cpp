#include "polish/progress.hpp"
#undef NDEBUG
#include <cassert>
int main(){
 hpr_polish::MilestoneProgress p;
 assert(!p.eligible(0,100,1e-4,1e-4,1e-3,1e-8));
 assert(p.eligible(700,100,1e-3,1e-3,.5,1e-8));
 p.launched(1e-3,1e-3,.5,1e-8);
 assert(!p.eligible(800,100,.1,.1,.5,1e-8));
 assert(p.eligible(900,100,1e-5,1e-5,.5,1e-8));
 assert(p.eligible(1000,100,1e-6,1e-6,.5,1e-8)); // busy milestones not consumed
 p.launched(1e-6,1e-6,.5,1e-8);
 assert(!p.eligible(1100,100,1e-5,1e-5,.5,1e-8));
 assert(p.eligible(1200,100,1e-7,1e-7,.5,1e-8));
 assert(!p.eligible(1300,100,1e-9,1e-9,1e-9,1e-8));
 p.launched(1e-7,1e-7,.5,1e-8);
 assert(p.eligible(1400,100,1e-8,1e-8,.5,1e-8));
 p.launched(1e-8,1e-8,.5,1e-8);
 assert(!p.eligible(1500,100,1e-8,1e-8,.5,1e-8));
 assert(p.eligible(1600,100,1e-12,1e-12,.5,1e-8));
 p.launched(0.,0.,.5,1e-8);
 assert(p.equation_target==0.);
 assert(!p.eligible(1700,100,0.,0.,.5,1e-8));
 assert(!p.eligible(1800,100,-1.,0.,.5,1e-8));
 assert(!p.eligible(1900,100,0.,0.,NAN,1e-8));
}
