#pragma once
// Read the independent CPU checker's certificate and bind it to the payload.
// No CUDA calls. OpenSSL's EVP SHA256 is loaded through its stable C interface.
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace hpr_polish {
inline std::array<unsigned char,32> payload_digest(const std::string& file){
    void* lib=dlopen("libcrypto.so.3",RTLD_NOW|RTLD_LOCAL);
    if(!lib)lib=dlopen("libcrypto.so.1.1",RTLD_NOW|RTLD_LOCAL);
    if(!lib)throw std::runtime_error("SHA256 provider unavailable");
    auto ctx_new=reinterpret_cast<void*(*)()>(dlsym(lib,"EVP_MD_CTX_new"));
    auto ctx_free=reinterpret_cast<void(*)(void*)>(dlsym(lib,"EVP_MD_CTX_free"));
    auto sha256=reinterpret_cast<const void*(*)()>(dlsym(lib,"EVP_sha256"));
    auto init=reinterpret_cast<int(*)(void*,const void*,void*)>(dlsym(lib,"EVP_DigestInit_ex"));
    auto update=reinterpret_cast<int(*)(void*,const void*,size_t)>(dlsym(lib,"EVP_DigestUpdate"));
    auto finish=reinterpret_cast<int(*)(void*,unsigned char*,unsigned int*)>(dlsym(lib,"EVP_DigestFinal_ex"));
    if(!ctx_new||!ctx_free||!sha256||!init||!update||!finish){dlclose(lib);throw std::runtime_error("SHA256 API unavailable");}
    void* ctx=ctx_new();std::array<unsigned char,32> digest{};
    try{
        if(!ctx||init(ctx,sha256(),nullptr)!=1)throw std::runtime_error("SHA256 init");
        std::ifstream f(file,std::ios::binary);if(!f)throw std::runtime_error("candidate missing");
        std::array<char,65536> buf{};
        while(f){f.read(buf.data(),buf.size());auto n=f.gcount();if(n&&update(ctx,buf.data(),size_t(n))!=1)throw std::runtime_error("SHA256 update");}
        if(!f.eof())throw std::runtime_error("candidate read failure");
        unsigned int n=0;if(finish(ctx,digest.data(),&n)!=1||n!=32)throw std::runtime_error("SHA256 final");
        ctx_free(ctx);dlclose(lib);return digest;
    }catch(...){if(ctx)ctx_free(ctx);dlclose(lib);throw;}
}
struct CpuCertificate {
    double tolerance,primal_feasibility,dual_feasibility,gap,primal_objective,dual_objective;
};
inline CpuCertificate read_cpu_certificate(const std::string& ep,const std::string& token,
                                           uint64_t n,uint64_t m,double tolerance){
    std::ifstream f(ep+"/certificate.bin",std::ios::binary);
    char magic[8];std::array<unsigned char,16> id{};uint64_t dims[2];
    std::array<unsigned char,32> expected{};double values[6];
    if(!f.read(magic,8)||!f.read((char*)id.data(),16)||!f.read((char*)dims,16)||
       !f.read((char*)expected.data(),32)||!f.read((char*)values,48)||f.peek()!=EOF)
        throw std::runtime_error("CPU certificate length");
    static const char* hex="0123456789abcdef";std::string actual;
    for(auto b:id){actual+=hex[b>>4];actual+=hex[b&15];}
    if(memcmp(magic,"HPRCPU01",8)||actual!=token||dims[0]!=n||dims[1]!=m)
        throw std::runtime_error("CPU certificate identity");
    for(double v:values)if(!std::isfinite(v))throw std::runtime_error("nonfinite CPU certificate");
    if(!(tolerance>0)||values[0]!=tolerance)throw std::runtime_error("CPU certificate tolerance");
    for(int i=1;i<=3;++i)if(values[i]<0||values[i]>tolerance)throw std::runtime_error("CPU certificate accuracy");
    // Also require consistency of the recorded objective values and gap.
    double gap=std::abs(values[4]-values[5])/(1+std::fmax(std::abs(values[4]),std::abs(values[5])));
    if(gap>tolerance||std::abs(gap-values[3])>1e-14)throw std::runtime_error("CPU certificate gap");
    if(n>INT32_MAX||m>INT32_MAX||std::filesystem::file_size(ep+"/candidate.bin")!=24+8*(3*n+2*m))
        throw std::runtime_error("CPU candidate length");
    if(payload_digest(ep+"/candidate.bin")!=expected)throw std::runtime_error("CPU certificate payload mismatch");
    return {values[0],values[1],values[2],values[3],values[4],values[5]};
}
} // namespace hpr_polish
