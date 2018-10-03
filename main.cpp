#include <iostream>
#include <blaze/Math.h>
#include <blaze/math/simd/SIMDPack.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>
#include <blaze/util/Complex.h>
#include <blaze/util/TrueType.h>
#include <blaze/math/blas/dotc.h>
#include <cstdlib>
#include <iostream>
#include <blaze/math/simd/BasicTypes.h>
#include <blaze/system/Inline.h>
#include <blaze/system/Vectorization.h>
#include <blaze/math/blas/dotc.h>
#include <blaze/math/Column.h>
//#include <BlazeIterative/BlazeIterative.hpp>
#include <immintrin.h>
#include <blaze/math/Band.h>

#include <blaze/math/constraints/SIMDPack.h>
using namespace blaze;
//using namespace blaze::iterative;
using blaze::randomize;
using blaze::CustomVector;
using blaze::unaligned;
using blaze::unpadded;
#include <blaze/math/BLAS.h>
#include <blaze/math/LowerMatrix.h>
#include <blaze/math/shims/Equal.h>
#include <blaze/math/StaticMatrix.h>
#include <blaze/math/StaticVector.h>
#include <blaze/math/UpperMatrix.h>
#include <blaze/math/blas/gemm.h>
#include <hpx/hpx_main.hpp>
void print_mat(blaze::DynamicMatrix<double> M)
{
    for (int i=0;i<M.rows();i++) {
        for (int j = 0; j < M.columns(); j++) {
            std::cout << M(i, j) << "   ";
        }
    std::cout<<std::endl;
    }
}


void print_vec(blaze::DynamicVector<double> V)
{
    for (int i=0;i<80;i++) {
            std::cout << V[i] << " ";
            }
            std::cout<<std::endl;
}

void print_vec(blaze::StaticVector<double, 7UL,blaze::rowVector> V)
{
    for (int i=0;i<7;i++) {
        std::cout << V[i] << "   ";
    }
}


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
        std::cout<<"None"<<std::endl;
    #endif

    size_t N(100000UL);
    blaze::Rand<blaze::DynamicVector<double>> gen{};

    blaze::DynamicVector<double>a =gen.generate( N );
    blaze::DynamicVector<double>b =gen.generate( N );
    blaze::DynamicVector<double>c =gen.generate( N );
    c=b+a;
    print_vec(a);
    print_vec(b);
    print_vec(c);
//    blaze::Rand<blaze::DynamicMatrix<double>> gen{};
//    blaze::DynamicMatrix<double> A=gen.generate(1000UL,1000UL);
//    blaze::DynamicMatrix<double> B=gen.generate(1000UL,1000UL);
//
//    blaze::DynamicMatrix<double> C;
//    C=A+3*B;
//    C=blaze::inv(A);

    return 0;
}


