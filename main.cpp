#include <iostream>
#include <algorithm>
#include <blaze/Math.h>
#include <cstdlib>
#include <iostream>
#include <boost/graph/graphviz.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <variant>
#include <string>
//#include <hpx/hpx_main.hpp>
#include <boost/graph/depth_first_search.hpp>
//#include <blaze/math/typetraits/getMflop.h>
using namespace blaze;


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

//    std::cout << "Using Boost "
//              << BOOST_VERSION / 100000     << "."  // major version
//              << BOOST_VERSION / 100 % 1000 << "."  // minor version
//              << BOOST_VERSION % 100                // patch level
//              << std::endl;
//

    std::size_t n = 913;
//    blaze::DynamicMatrix<double, true> C(n, n, 2.0);
    blaze::DynamicMatrix<double> B(n, n, 3.0);
//    blaze::DynamicMatrix<double> D(n, n, 0.0);
//    blaze::DynamicMatrix<double, true> E(n, n, 4.0);
//    blaze::DynamicVector<double> VB(1000000, 3.0);

    blaze::DynamicVector<double> V1(24, 2.0);
    std::iota(V1.begin(),V1.end(),0.);
    std::cout<<V1<<std::endl;
    blaze::CustomMatrix<double,blaze::unaligned,blaze::unpadded> tmp(&V1[0],4,6);
    std::cout<<tmp<<std::endl;
    auto k = B * B ;
//    std::cout<< typeid(k).name()<<std::endl;
    B = k;
//    blaze::DynamicMatrix<int, blaze::rowMajor> M{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
//    blaze::DynamicMatrix<int, blaze::columnMajor> Mt(M);

//    std::cout << blaze::getTotalMflop(M * M * M) << std::endl;         //  90 ok
//    std::cout << blaze::getTotalMflop(Mt * Mt * Mt) << std::endl;      //  90 ok
//    std::cout << blaze::getTotalMflop(M * (M + M)) << std::endl;       //  54 ok
//    std::cout << blaze::getTotalMflop((M + M) * M) << std::endl;       //  54 ok
//    std::cout << blaze::getTotalMflop((M + M) * (M + M)) << std::endl; //  63
//    std::cout << blaze::getTotalMflop(M * M + M * M) << std::endl;     //  9 not ok should be 99 !!!
//    std::cout << blaze::getTotalMflop(5 * M + M * M) << std::endl;     //  63
    return 0;
}

