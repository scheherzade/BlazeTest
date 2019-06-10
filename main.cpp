#include <iostream>
#include <algorithm>
#include <blaze/Math.h>
#include <cstdlib>
#include <iostream>

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


    std::size_t n = 1000;
    blaze::DynamicMatrix<double> C(n, n, 2.0);
    blaze::DynamicMatrix<double> B(n, n, 3.0);
    blaze::DynamicMatrix<double> D(n, n, 0.0);
    blaze::DynamicMatrix<double> E(n, n, 0.0);

    auto k = B * C;
    D = k;

    auto m = B + C;
    E = m;	

    return 0;
}


