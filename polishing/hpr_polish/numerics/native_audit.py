"""Structural numerical primitives; no experiment runner or standalone optimizer."""
import numpy as np


def replay(mp,state,direction=None):
    m=mp.m;n=mp.n;l=m.linear_m;dtype=np.longdouble
    x,w,y=np.split(np.asarray(state,dtype=dtype),[n,2*n])
    sig=dtype(mp.sigma);tau=dtype(mp.tau);q=m.q.astype(dtype);D=1/(1+sig*q)
    raw=x+sig*(m.A.T@y-q*w-m.c);p=np.minimum(np.maximum(raw,m.lo),m.hi)
    z=(p-raw)/sig;xh=2*p-x;w1=D*(sig*q*w+xh)
    temp=xh+sig*q*(w-w1);v=m.A@temp-m.rhs-tau*y
    s=np.empty_like(v);s[:l]=np.minimum(np.maximum(v[:l],m.al),m.au)
    vc=v[l:].reshape(-1,3);sc=s[l:].reshape(-1,3);r=np.sqrt(np.sum(vc[:,1:]**2,axis=1));t=vc[:,0]
    inside=r<=t;outside=(r<=-t)&~inside;middle=~(inside|outside)
    sc[:]=0;sc[inside]=vc[inside];sc[middle,0]=(r[middle]+t[middle])/2
    sc[middle,1:]=((r[middle]+t[middle])/(2*r[middle]))[:,None]*vc[middle,1:]
    yb=(s-v)/tau;wb=w1+sig*D*(m.A.T@(yb-y));F=np.r_[x-xh,2*(w-wb),2*(y-yb)]
    Qp=q*p;Ap=m.A@p;Aty=m.A.T@yb
    primal=np.sum(m.c*p)+np.sum(p*Qp)/2+m.constant
    dual=-np.sum(p*Qp)/2+np.sum(z*p)+np.sum(yb*(s+m.rhs))+m.constant
    gap=abs(primal-dual)/(1+max(abs(primal),abs(dual)))
    rd=max(abs(Qp+m.c-Aty-z))/(1+max(max(abs(m.c)),max(abs(Qp)),max(abs(Aty))))
    vv=Ap-m.rhs;linear=max(abs(np.clip(vv[:l],m.al,m.au)-vv[:l]),default=0)
    cv=vv[l:].reshape(-1,3);cr=np.sqrt(np.sum(cv[:,1:]**2,axis=1));ct=cv[:,0]
    ci=cr<=ct;co=(cr<=-ct)&~ci;cm=~(ci|co);proj=np.zeros_like(cv);proj[ci]=cv[ci]
    proj[cm,0]=(cr[cm]+ct[cm])/2;proj[cm,1:]=((cr[cm]+ct[cm])/(2*cr[cm]))[:,None]*cv[cm,1:]
    rp=max(linear,np.max(abs(proj-cv)))/(1+max(mp.bnorm,max(abs(Ap))))
    norm=float(np.sqrt(np.sum(F*F)))
    ans=dict(F=F,norm=norm,certificate=dict(error=float(max(rp,rd,gap)),rp=float(rp),rd=float(rd),gap=float(gap),primal=float(primal),dual=float(dual)))
    if direction is not None:
        dx,dw,dy=np.split(np.asarray(direction,dtype=dtype),[n,2*n]);dr=dx+sig*(m.A.T@dy-q*dw)
        dp=((raw>m.lo)&(raw<m.hi))*dr;dxh=2*dp-dx;dw1=D*(sig*q*dw+dxh)
        dv=m.A@(dxh+sig*q*(dw-dw1))-tau*dy;ds=np.zeros_like(dv)
        ds[:l]=((v[:l]>m.al)&(v[:l]<m.au))*dv[:l]
        dc=dv[l:].reshape(-1,3);dsc=ds[l:].reshape(-1,3);dsc[inside]=dc[inside]
        u=vc[middle,1:]/r[middle,None];ud=np.sum(u*dc[middle,1:],axis=1)
        dsc[middle,0]=(dc[middle,0]+ud)/2
        dsc[middle,1:]=u*(dc[middle,0]/2)[:,None]+((1+t[middle]/r[middle])/2)[:,None]*dc[middle,1:]-(t[middle]*ud/(2*r[middle]))[:,None]*u
        dyb=(ds-dv)/tau;dwb=dw1+sig*D*(m.A.T@(dyb-dy))
        Jd=np.r_[dx-dxh,2*(dw-dwb),2*(dy-dyb)]
        ans['linear_defect']=float(np.sqrt(np.sum((Jd+F)**2)))/norm
        ans['Jd']=Jd
    return ans
