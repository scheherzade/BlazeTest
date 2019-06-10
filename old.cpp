//#include <iostream>
//#include <algorithm>
//#include <blaze/Math.h>
//#include <blaze/math/simd/SIMDPack.h>
//#include <blaze/system/Inline.h>
//#include <blaze/system/Vectorization.h>
//#include <blaze/util/Complex.h>
//#include <blaze/util/TrueType.h>
//#include <blaze/math/blas/dotc.h>
//#include <cstdlib>
//#include <iostream>
//#include <blaze/math/simd/BasicTypes.h>
//#include <blaze/system/Inline.h>
//#include <blaze/system/Vectorization.h>
//#include <blaze/math/blas/dotc.h>
//#include <blaze/math/Column.h>
////#include <BlazeIterative/BlazeIterative.hpp>
//#include <immintrin.h>
//#include <blaze/math/Band.h>
////#include <blaze_tensor/Math.h>
//
//#include <blaze/math/constraints/SIMDPack.h>
//using namespace blaze;
////using namespace blaze::iterative;
//using blaze::randomize;
//using blaze::CustomVector;
//using blaze::unaligned;
//using blaze::unpadded;
//#include <blaze/math/BLAS.h>
//#include <blaze/math/LowerMatrix.h>
//#include <blaze/math/shims/Equal.h>
//#include <blaze/math/StaticMatrix.h>
//#include <blaze/math/StaticVector.h>
//#include <blaze/math/UpperMatrix.h>
//#include <blaze/math/blas/gemm.h>
//#include <variant>
//#include <string>
//#include <hpx/hpx_main.hpp>
//
//void print_mat(blaze::DynamicMatrix<double> M)
//{
//    for (int i=0;i<M.rows();i++) {
//        for (int j = 0; j < M.columns(); j++) {
//            std::cout << M(i, j) << "   ";
//        }
//    std::cout<<std::endl;
//    }
//}
//
//
//void print_vec(blaze::DynamicVector<double> V)
//{
//    for (int i=0;i<V.size();i++) {
//            std::cout << V[i] << " ";
//            }
//            std::cout<<std::endl;
//}
//
//void print_vec(blaze::DynamicVector<double, blaze::rowVector> V)
//{
//    for (int i=0;i<V.size();i++) {
//        std::cout << V[i] << " ";
//    }
//    std::cout<<std::endl;
//}
//
////void print_vec(blaze::StaticVector<double, 7UL,blaze::rowVector> V)
////{
////    for (int i=0;i<7;i++) {
////        std::cout << V[i] << "   ";
////    }
////}
//
//bool test(blaze::DynamicVector<double>* lhs, blaze::DynamicVector<double>* rhs)
//{
//    return std::lexicographical_compare(lhs->data(), lhs->data() + (*lhs).size(),
//                                        (rhs)->data(), rhs->data() + (*rhs).size());
//}
//int gcd(int a, int b)
//{
//    int result = 1;
//    if (a==1)
//        return result;
//    for (int i = 2; i <= a; ++i)
//    {
//        if (a%i == 0 && b%i == 0)
//        {
//            result = i;
//        }
//    }
//    return result;
//}
//int main() {
//
//
//#if defined(BLAZE_USE_HPX_THREADS)
//    std::cout<<"hpx"<<std::endl;
//#endif
//
//#if BLAZE_OPENMP_PARALLEL_MODE
//    std::cout<<"BLAZE_OPENMP_PARALLEL_MODE"<<std::endl;
//#elif BLAZE_CPP_THREADS_PARALLEL_MODE || BLAZE_BOOST_THREADS_PARALLEL_MODE
//    std::cout<<"BLAZE_Boost_CPP_PARALLEL_MODE"<<std::endl;
//#elif BLAZE_HPX_PARALLEL_MODE
//    std::cout<<"BLAZE_HPX_PARALLEL_MODE"<<std::endl;
//#else
//    std::cout << "None" << std::endl;
//#endif
//std::cout<<BLAZE_SMP_DMATDMATADD_THRESHOLD<<std::endl;
////    std::cerr << "\n"
////              << " BLAZE_SSE_MODE      = " << BLAZE_SSE_MODE << "\n"
////              << " BLAZE_SSE2_MODE     = " << BLAZE_SSE2_MODE << "\n"
////              << " BLAZE_SSE3_MODE     = " << BLAZE_SSE3_MODE << "\n"
////              << " BLAZE_SSSE3_MODE    = " << BLAZE_SSSE3_MODE << "\n"
////              << " BLAZE_SSE4_MODE     = " << BLAZE_SSE4_MODE << "\n"
////              << " BLAZE_AVX_MODE      = " << BLAZE_AVX_MODE << "\n"
////              << " BLAZE_AVX2_MODE     = " << BLAZE_AVX2_MODE << "\n"
////              << " BLAZE_AVX512F_MODE  = " << BLAZE_AVX512F_MODE << "\n"
////              << " BLAZE_AVX512BW_MODE = " << BLAZE_AVX512BW_MODE << "\n"
////              << " BLAZE_AVX512DQ_MODE = " << BLAZE_AVX512DQ_MODE << "\n"
////              << " BLAZE_MIC_MODE      = " << BLAZE_MIC_MODE << "\n"
////              << "\n"
////              << " blaze::Abs::simdEnabled<int>() = " << blaze::Destroy::simdEnabled<int>() << "\n"
////              << std::endl;
//
////    blaze::DynamicVector<std::int64_t> decoded{1,1,0,4,2,2,3,1};
////    std::size_t num_classes=3;
////    auto tmp1 = blaze::subvector(decoded, 1, decoded.size()-1);
////    std::cout<<decoded<<std::endl;
////    auto tmp2 = blaze::subvector(decoded, 0, decoded.size()-1);
////    std::cout<<decoded<<std::endl;
////double* it;
////*it =1;
////*(it+1)=2;
////    CustomVector<double,unaligned,unpadded> A( it, 2UL );
////    std::cout<<A<<std::endl;
////    auto tmp3 = blaze::subvector(decoded, 0, decoded.size()-1);
////    std::cout<<decoded<<std::endl;
//
////    blaze::DynamicVector<std::int64_t>tmp3(decoded.size());
////    tmp3= tmp2 - tmp1;
////
////    blaze::DynamicVector<std::int64_t> d(decoded.size(),0);
////    std::size_t k = 0;
////    for (std::size_t j = 0; j < decoded.size() - 1 ; ++j)
////    {
////        if (decoded[j] != decoded[j + 1]) {
////            if (decoded[j] < num_classes-1)
////            decoded[k++] = decoded[j];
////
////        }
////    }
////    decoded[k++] = decoded[decoded.size()-1];
////    decoded.resize(k);
////    d.resize(k);
////    std::cout<<"result"<<decoded<<std::endl;
//
////    blaze::DynamicTensor<double> t{{{0.2, 0.2, 0.6},{0.4, 0.3, 0.3}},{{0.7, 0.15, 0.15},{0., 0., 0.}}};
////    auto a = blaze::rowslice(t,0);
////
////    blaze::DynamicMatrix<double> ravel_test{{1, 2}, {3, 4}};
////    blaze::DynamicVector<double> test{5, 6, 7, 8};
////    auto x = blaze::ravel(ravel_test);
////    std::cout<<"rowslice: "<<blaze::rowslice(t, 0)<<std::endl;
//
////    blaze::DynamicTensor<double> t{{{7.0, 8.0, 11.0}, {9.0, 10., 12.0}}, {{3.0, 4.0, 5.0}, {3.0, 4.0, 5.0}}};
////    blaze::DynamicTensor<double> result(t.pages()* 2, t.rows() * 3, t.columns() );
//
////    for (std::size_t i = 0; i != result.rows(); ++i)
////    {
////        for (std::size_t j = 0; j != result.pages(); ++j) {
////            blaze::column(blaze::rowslice(result, i), j) =
////                    blaze::column(blaze::rowslice(t, static_cast<std::int64_t>(i / 3)), static_cast<std::int64_t>(j / 2));
////        }
////    }
////
////    std::cout<<"t: "<<t<<std::endl;
////    std::cout<<"result: "<<result<<std::endl;
////
////    std::cout<<"pageslice: "<<blaze::row(blaze::pageslice(t, 0),0)<<std::endl;
//    std::size_t n = 1000;
//    blaze::DynamicMatrix<double> CC(n, n, 2.0);
//    blaze::DynamicMatrix<double> BB(n, n, 3.0);
//    blaze::DynamicMatrix<double> DD(n, n, 0.0);
//    blaze::DynamicVector<double> V(n, 6), M(n,0);
////    auto dsm4 = submatrix<aligned>( CC, 2UL, 3UL, 12UL, 12UL );
//
//        auto k = BB * CC;
//        DD=k;
//
////std::cout<<DD<<std::endl;
////    auto s(submatrix(CC,0UL,0UL,1UL,10UL));
////    auto ss(submatrix(BB,0UL,0UL,10UL,1UL));
//////    auto sss(submatrix(DD,0UL,0UL,1UL,1UL));
////
////    blaze::DynamicMatrix<double> D(64UL, 64UL, 2);
////    submatrix(DD,0UL,0UL,1UL,1UL)= s*ss;
////
////    std::cout<<DD<<std::endl;
//////    for (int i=0UL;i<10UL;i++)
//////    {
//////        auto s(submatrix(CC,i,0UL,1UL,10UL));
//////
//////    }
////    std::array<std::size_t, 3> sizes{{0,0,0}};
////
////    blaze::DynamicMatrix<double> C(4UL, 4UL, 3);
////blaze::column(C,0)=2;
////std::cout<<C<<std::endl;
////    blaze::DynamicVector<double> v1{9,5,8};
////    blaze::DynamicVector<double> v111{0,1};
////    std::vector<std::size_t> test(2,0);
////
////    std::vector<std::size_t> ind{1,2};
////    auto e = elements(v1, v111.data(), v111.size());
////    blaze::DynamicVector<double> result(1+v1.size(), 1.0);
////
////    std::copy(v1.begin(), v1.end(), result.begin());
////
////    blaze::DynamicVector<double> v11(v1.size());
////    std::iota(v11.begin(), v11.end(), 0);
////    std::sort(v11.begin(), v11.end(), [&](double a, double b){return v1[a]<v1[b];});
//
////    std::vector<double> T{1,2,3,4,5};
////    double global_max = std::numeric_limits<double>::max();
////    double global_min = std::numeric_limits<double>::min();
////    blaze::DynamicMatrix<double> m{{1,2,3},{6,1,9}};
////    blaze::DynamicMatrix<double> result(4,9,0.);
////    std::size_t height_factor=2;
////    std::size_t width_factor=3;
////
////    for (std::size_t i = 0; i < m.rows(); ++i) {
////        for (std::size_t j = 0; j < m.columns(); ++j) {
////            for (std::size_t k = 0; k < width_factor; ++k) {
////
////                result(height_factor * i, width_factor * j + k) =
////                        m(i, j) + k * (m(i, j + 1) - m(i, j)) / width_factor;
////
////
////            }
////        }
////    }
//

////
//////    for (std::size_t i = 0; i != result.rows(); ++i) {
//////        auto s = blaze::row(result , i);
//////
//////        auto e1 = elements(s, [&](size_t j) { return j * width_factor; }, m.columns());
//////        e1 = blaze::row(m, i/height_factor);
//////    }
////
////    blaze::DynamicMatrix<double> tmp(2UL,3UL,-1.);
////    tmp=blaze::abs(tmp);
////    std::cout<<tmp<<std::endl;
//
////    auto rs1 = rows( mv1, []( size_t i ){ return i*2UL; }, 1UL );
////    auto rs2 = columns( rs1, []( size_t i ){ return i*2UL; }, 1UL );
////
////    blaze::DynamicMatrix<double> tmp(1UL,1UL,0.);
////    rs2=tmp;
////    std::cout<<tmp<<std::endl;
//
////    blaze::DynamicMatrix<double> mv3{{0,4,2},{0,4,2}};
////    blaze::DynamicMatrix<double> mv2{{1,1,1},{1,1,1}};
////auto result1=blaze::map(mv1, mv2,
////                        [&](double x, double y) { return blaze::clamp(x, y, global_max); });
////    auto result2 = blaze::map(result1,
////            mv3, [&](double x, double y) { return blaze::clamp(x, global_min, y); });
////    std::cout<<result1<<std::endl;
////
////    std::cout<<result2<<std::endl;
////
////    auto d= mv1.data();
//
//
//
//
//    //blaze::DynamicMatrix<double> mv1{v1.size(),1,v1.data()};
//
//
//
//
////    print_vec(v1);
////    auto bb=v1.data();
////
////
////    blaze::DynamicVector<double> v2{11, 12, 4};
////
////    blaze::DynamicVector<double> v3(4UL, 1.0);
////print_vec(v3);
////
////    blaze::DynamicVector<double>* x= &v1;
////    if (test(&v1,&v2))
////        std::cout<<"yes"<<std::endl;
////    blaze::DynamicMatrix<double> m{{1,2,5},{3,4,9},{8,3,1}};
//////    m=blaze::map(m, [](double a){return blaze::clamp(a, 5.0, 12.0);});
////    m=blaze::clamp(m, 5.0, 12.0);
////    print_mat(m);
//////    std::transform(m.data(), m.data()+m.rows()*m.columns(), m.data(), [](double a)->double{return std::clamp(a, 5.0, 12.0);});
////
////    print_mat(m);
////    blaze::DynamicMatrix<double> n(5UL,2UL,3.0);
////
////    auto band1=blaze::band(n,1L);
////
//////    band1=v3;
////
////    print_mat(n);
////    std::cout<<std::endl;
////    print_vec(v1);
////    std::cout<<std::endl;
////    v1.resize(3UL);
////    print_vec(v1);
////    std::cout<<std::endl;
////    size_t N(3UL);
////    blaze::Rand<blaze::DynamicVector<double>> gen{};
////
////    blaze::DynamicVector<double>a =gen.generate( N );
////    blaze::DynamicVector<double>b =gen.generate( N );
////    blaze::DynamicVector<double>c =gen.generate( N );
////
////    size_t num_users(5UL);
////    size_t num_items(10UL);
////    size_t num_factors(3UL);
////
//////    c=b+a;
////    blaze::Rand<blaze::DynamicMatrix<double>> gen_m{};
////    blaze::DynamicMatrix<double> B(600UL, 600UL, 1.5);
////
////
////
////    std::cout<<blaze::pow(mv1,4)<<std::endl;
////    blaze::DynamicMatrix<double> A(600UL, 600UL, 2);
//
////    blaze::DynamicMatrix<double> D(64UL, 64UL, 2);
//
////    std::cout<<blaze::map(v1, [&](double x){return blaze::clamp(C, 1.0, x);});
////    print_vec(v1);
////    print_mat(A);
////    using var_t=std::variant<double, std::string>;
////    std::vector<var_t> tmp;
//
////    std::cout<<blaze::trace(A)<<std::endl;
////    print_vec(a);
////    print_vec(b);
////    print_vec(c);
////    blaze::Rand<blaze::DynamicMatrix<double>> gen{};
////    blaze::DynamicMatrix<double> A=gen.generate(1000UL,1000UL);
////    blaze::DynamicMatrix<double> B=gen.generate(1000UL,1000UL);
////
////    blaze::DynamicMatrix<double> C;
////    C=A+3*B;
////    C=blaze::inv(A);
//
//    return 0;
//}
//
//
//
////using CustomType = CustomVector<int,unaligned,unpadded>;
////struct div_simd
////{
////    template< typename T >
////    BLAZE_ALWAYS_INLINE auto operator()( const T& a ,const T& b) const -> T
////    {
////        return a/b;
////    }
////
////    template< typename T1,typename T2 >
////    static constexpr bool simdEnabled() { return blaze::HasSIMDDiv<T1,T2>::value; }
////
////    template< typename T >
////    BLAZE_ALWAYS_INLINE auto load( const T& a ,const T& b) const ->T
////    {
////        BLAZE_CONSTRAINT_MUST_BE_SIMD_PACK( T );
////        return a/b;
////
////    }
////};
//////
//////    __m256 evens = _mm256_set_ps(2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0);
//////    __m256 odds = _mm256_set_ps(1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0);
//////
//////    /* Compute the difference between the two vectors */
//////    __m256 result = _mm256_add_ps(evens, odds);
//////
//////    /* Display the elements of the result vector */
//////    float* f = (float*)&result;
//////    printf("%f %f %f %f %f %f %f %f\n",
//////           f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]);
////
////
////
//////    std::size_t vector_size=4UL;
//////    std::size_t rows,cols;
//////    rows=2UL;
//////    cols=4UL;
//////    blaze::DynamicVector<double>v1{0,1,2,3};
//////    v1 = blaze::normalize( v1 );
//////    for (int i=0;i<v1.size();i++) {
//////        std::cout << v1[i] << ",";
//////    }
//////    std::cout<<std::endl;
//////
//////    std::vector<int> vec( 5UL, 10 );  // Vector of 5 integers of the value 10
//////    CustomType a( &vec[0], 5UL );     // Represent the std::vector as Blaze dense vector
//////    vec[1] = 20;                        // Also modifies the std::vector
//////
//////    CustomType b( a );  // Creating a copy of vector a
//////    b[2] = 20;
//////
//////    for (int i=0;i<vec.size();i++) {
//////        std::cout << vec[i] << ",";
//////    }
//////    std::cout<<std::endl;
//////
//////    for (int i=0;i<a.size();i++) {
//////        std::cout << a[i] << ",";
//////    }
//////    std::cout<<std::endl;
//////
//////    for (int i=0;i<b.size();i++) {
//////        std::cout << b[i] << ",";
//////    }
////
//////    blaze::DynamicVector<double> A1 {0,1,2,3};
//////    blaze::DynamicVector<bool> A2 {true, false, false, true};
//////    A1=A1+A2;
////
////
//////    auto x=static_cast<bool> (1);
//////
//////    blaze::Rand<blaze::DynamicVector<double>> gen{};
//////    blaze::DynamicVector<double> b {0,1,2,3};
//////    blaze::DynamicVector<double> A1 {0,1,2,3};
//////    blaze::DynamicVector<double> A2 {0,1,2,3};
//////    A=blaze::map(A1,A2,[](double x, double y){return x==y;});
////
//////    blaze::DynamicVector<double> B {9,1};
//////    blaze::DynamicVector<double> C (vector_size);
//////
//////    blaze::Rand<blaze::DynamicMatrix<double>> mat_gen{};
//////    blaze::DynamicMatrix<double> A {{1,2,6,9},{3,1,7,2}};
//////    blaze::DynamicMatrix<double> m2 = mat_gen.generate(rows,cols);
////
////
////constexpr size_t N( 2UL );
////
////blaze::DynamicMatrix<double,blaze::columnMajor> A{{2,1},{1,-1}};
////// ... Initializing the matrix
////
////const int m    ( blaze::numeric_cast<int>( A.rows()    ) );  // == N
////const int n    ( blaze::numeric_cast<int>( A.columns() ) );  // == N
////const int lda  ( blaze::numeric_cast<int>( A.spacing() ) );  // >= N
////const int lwork( n*lda );
////
////const std::unique_ptr<int[]> ipiv( new int[N] );        // No initialization required
////const std::unique_ptr<double[]> work( new double[N] );  // No initialization required
////
////int info( 0 );
////
////blaze::getrf( m, n, A.data(), lda, ipiv.get(), &info );
////print_mat(A);
//////
//////    for (int i=0;i<rows;i++) {
//////        for (int j = 0; j < cols; j++)
//////            std::cout << m1(i, j) << ",";
//////        std::cout << std::endl;
//////    }
//////    std::cout << std::endl;
//////    for (int i=0;i<vector_size;i++)
//////        std::cout<<A[i]<<",";
//////    std::cout<<std::endl;
//////
//////    for (size_t i=0UL;i<m1.rows();i++)
//////        blaze::row(m2,i)=blaze::row(m1,i)*blaze::trans(A);
//////
//////
//////
//////    std::cout << std::endl;
//////    for (int i=0;i<rows;i++) {
//////        for (int j = 0; j < cols; j++)
//////            std::cout << m2(i, j) << ",";
//////        std::cout << std::endl;
//////    }
//////    std::cout <<std::endl<< "colwise"<<std::endl;
//////    m1 ={{1,2,6,9},{3,1,7,2}};
//////
//////    for (int i=0;i<rows;i++) {
//////        for (int j = 0; j < cols; j++)
//////            std::cout << m1(i, j) << ",";
//////        std::cout << std::endl;
//////    }
//////
//////    std::cout << std::endl;
//////    for (int i=0;i<2UL;i++)
//////        std::cout<<B[i]<<",";
//////    std::cout<<std::endl;
//////
//////    blaze::DynamicMatrix<double,columnMajor> m4=blaze::trans(m1);
//////    for (size_t i=0UL;i<m1.columns();i++)
//////        column(m1,i)+=B;
//////
//////    for (int i=0;i<rows;i++) {
//////        for (int j = 0; j < cols; j++)
//////            std::cout << m1(i, j) << ",";
//////        std::cout << std::endl;
//////    }
////
////
////constexpr size_t N( 100UL );
//////    blaze::DynamicMatrix<double, columnMajor> A{{3,1,-1},{2,-1,1},{-1,3,-2}};
//////    blaze::DynamicMatrix<double, columnMajor> A{{4,1},{1,2}};
////blaze::DynamicMatrix<double, columnMajor> A{{7,7,10,8},{8,2,3,3},{3,1,6,7},{7,5,2,7}};
////blaze::DynamicMatrix<double> B{{7,7,10,8},{8,2,3,3},{3,1,6,7},{7,5,2,7}};
////blaze::DynamicMatrix<double> C=blaze::trans(B);
////
//////std::cout<<blaze::min(3,5)<<std::endl;
////
////print_mat(A);
////std::cout<<std::endl;
////// ... Initializing the matrix
//////    blaze::DynamicVector<double> b{2, 3, -1};
//////    blaze::DynamicVector<double> b{0, 0, 1};
////
////
//////    const std::unique_ptr<double[]> ipiv( new double[blaze::min(A.rows(),A.columns())] );  // No initialization required
////const std::unique_ptr<int[]> ipiv( new int[blaze::min(A.rows(),A.columns())] );  // No initialization required
////
////
//////    blaze::sysv( A, b, 'L', ipiv.get());
//////    blaze::posv( A, b, 'L');
////std::cout<<"pivot points"<<std::endl;
////for (int i=0;i<blaze::min(A.rows(),A.columns());i++) {
////std::cout << ipiv[i] << "   ";
////}
////std::cout<<std::endl;
////blaze::getrf(A, ipiv.get());
////std::cout<<"pivot points"<<std::endl;
////
////for (int i=0;i<blaze::min(A.rows(),A.columns());i++) {
////std::cout << ipiv[i] << "   ";
////}
////std::cout<<std::endl<<std::endl;
////
////print_mat(A);
////std::cout<<std::endl<<"B"<<std::endl;
////const std::unique_ptr<int[]> ipivB(new int[blaze::min(B.rows(),B.columns())] );  // No initialization required
////
////
//////    blaze::sysv( A, b, 'L', ipiv.get());
//////    blaze::posv( A, b, 'L');
////std::cout<<"pivot points"<<std::endl;
////for (int i=0;i<blaze::min(B.rows(),B.columns());i++) {
////std::cout << ipivB[i] << "   ";
////}
////std::cout<<std::endl;
////blaze::getrf(B, ipivB.get());
////std::cout<<"pivot points"<<std::endl;
////
////for (int i=0;i<blaze::min(B.rows(),B.columns());i++) {
////std::cout << ipivB[i] << "   ";
////}
////std::cout<<std::endl<<std::endl;
////
////print_mat(B);
////std::cout<<std::endl;
//////    print_vec(b);
////
////std::cout<<std::endl<<"C"<<std::endl;
////const std::unique_ptr<int[]> ipivC(new int[blaze::min(C.rows(),C.columns())] );  // No initialization required
////
////
//////    blaze::sysv( A, b, 'L', ipiv.get());
//////    blaze::posv( A, b, 'L');
////std::cout<<"pivot points"<<std::endl;
////for (int i=0;i<blaze::min(C.rows(),C.columns());i++) {
////std::cout << ipivC[i] << "   ";
////}
////std::cout<<std::endl;
////blaze::getrf(C, ipivC.get());
////std::cout<<"pivot points"<<std::endl;
////
////for (int i=0;i<blaze::min(C.rows(),C.columns());i++) {
////std::cout << ipivC[i] << "   ";
////}
////std::cout<<std::endl<<std::endl;
////
////print_mat(C);
////std::cout<<std::endl;
//////    print_vec(b);
////
////
////
//////    std::cout<<m1(2,1)<<","<<A[1]<<": ";
//////    m3=blaze::trans(m1);
//////    column(m3, 2UL )+=(A);
//////    C=blaze::map(A,B,div_simd());
//////C=blaze::map(A,B,[](double x, double y){return x/y;});
