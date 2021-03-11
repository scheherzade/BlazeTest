#include <iostream>
#include <algorithm>
#include <blaze/Math.h>
//#include <blaze_tensor/Math.h>
#include <cstdlib>
#include <iostream>
#include <blaze/math/traits/AddTrait.h>
#include <blaze/math/smp/hpx/cuda/util/CUDAAllocator.h>
using namespace blaze;
#include <variant>
#include <string>
#include <hpx/hpx_main.hpp>

int main() {

#if defined(BLAZE_USE_HPX_THREADS)
    std::cout<<"hpx"<<std::endl;
#endif

#if BLAZE_OPENMP_PARALLEL_MODE
    std::cout<<"BLAZE_OPENMP_PARALLEL_MODE"<<std::endl;
#elif BLAZE_CPP_THREADS_PARALLEL_MODE || BLAZE_BOOST_THREADS_PARALLEL_MODE
    std::cout<<"BLAZE_Boost_CPP_PARALLEL_MODE"<<std::endl;
#elif BLAZE_HPX_PARALLEL_MODE
    std::cout<<"BLAZE_HPX_PARALLEL_MODE"<<std::endl;
#else
    std::cout << "None" << std::endl;
#endif
    using blaze::AlignedAllocator;
//    using Group11 = blaze::GroupTag<11>;
    using CUDAGroup = blaze::GroupTag<9>;
//    using blaze::CUDAAllocator;


    std::size_t n = 600;
#if 1
    #ifndef __CUDACC_EXTENDED_LAMBDA__
    #error "please compile with --expt-extended-lambda"
    #endif

    blaze::DynamicMatrix<double, rowMajor, blaze::CUDAAllocator<double>, CUDAGroup> C(n, n, 3.0);
    blaze::DynamicMatrix<double, rowMajor, blaze::CUDAAllocator<double>, CUDAGroup> Q(n, n, 3.0);


    blaze::DynamicMatrix<double, rowMajor, blaze::CUDAAllocator<double>, CUDAGroup> H = C + Q;
#else
//
    blaze::DynamicMatrix<double,rowMajor, blaze::AlignedAllocator<double>, blaze::GroupTag<9>> B(n, n, 3.0);
    blaze::DynamicMatrix<double,rowMajor, blaze::AlignedAllocator<double>, blaze::GroupTag<9>> D(n, n, 0.0);
    blaze::DynamicMatrix<double,rowMajor, blaze::AlignedAllocator<double>, blaze::GroupTag<9>> E(n, n, 2.0);
    auto k = B + E;
    D = k;
#endif
//    blaze::DynamicTensor<double> a{{{1,2,3,4},{5,6,7,8}},{{9,10,11,12},{13,14,15,16}},{{17,18,19,20},{21,22,23,24}}};
//    blaze::DynamicTensor<double> b{{{20,19},{18,17}},{{16,15},{14,13}},{{12,11},{10,9}},{{8,7},{6,5}},{{4,3},{2,1}}};
//
//    blaze::DynamicTensor<double> c(a.pages(),b.pages(),a.columns()-b.columns()+1,0);

//    for (std::size_t p=0;p<a.pages();++p)
//    {
//        for (std::size_t l=0;l<b.pages();++l)
//        {
//            for (std::size_t m=0;m<a.columns()-b.columns()+1;++m)
//            {
//                std::cout<<p<<","<<l<<","<<m<<std::endl;
//                auto x = blaze::subtensor(a, p, 0, m, 1, a.rows(), b.columns());
//                auto y = blaze::subtensor(b, l, 0, 0, 1, b.rows(), b.columns());
//                auto w = blaze::sum(x%y);
//                blaze::subtensor(c, p, l, m, 1, 1, 1) = w;
//            }
//        }
//    }
////    auto c = blaze::pageslice(a,0);
//    auto d = blaze::rowslice(b,0);
//
//    auto e = blaze::subtensor(a,1,0,0,1,a.rows(),a.columns());
//    auto f = blaze::subtensor(b,0,0,0,2,1,1);

//    auto g = a % b;
//    std::cout<<c<<std::endl;
//    std::cout<<c.pages()<<","<<c.rows()<<","<<c.columns()<<std::endl;
//    std::cout<<b.pages()<<","<<b.rows()<<","<<b.columns()<<std::endl;

//    std::cout<<c.rows()<<","<<c.columns()<<","<<c.page()<<std::endl;
//    std::cout<<d.rows()<<","<<d.column()<<","<<d.pages()<<std::endl;



//    std::cout<<D<<std::endl;

    return 0;
}


