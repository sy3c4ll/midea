# cmake_minimum_require must be the first command of the file
cmake_minimum_required(VERSION 3.5.0)

project(Eigen3)

set(CMAKE_CXX_STANDARD 11 CACHE STRING "Default C++ standard")
set(CMAKE_CXX_STANDARD_REQUIRED ON CACHE BOOL "Require C++ standard")
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL "Allow C++ extensions")

macro(ei_add_cxx_compiler_flag FLAG)
  string(REGEX REPLACE "-" "" SFLAG1 ${FLAG})
  string(REGEX REPLACE "\\+" "p" SFLAG ${SFLAG1})
  check_cxx_compiler_flag(${FLAG} COMPILER_SUPPORT_${SFLAG})
  if(COMPILER_SUPPORT_${SFLAG})
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
  endif()
endmacro()

include(CheckCXXCompilerFlag)

if(NOT MSVC)
  # We assume that other compilers are partly compatible with GNUCC

  # clang outputs some warnings for unknown flags that are not caught by check_cxx_compiler_flag
  # adding -Werror turns such warnings into errors
  check_cxx_compiler_flag("-Werror" COMPILER_SUPPORT_WERROR)
  if(COMPILER_SUPPORT_WERROR)
    set(CMAKE_REQUIRED_FLAGS "-Werror")
  endif()
  ei_add_cxx_compiler_flag("-pedantic")
  ei_add_cxx_compiler_flag("-Wall")
  ei_add_cxx_compiler_flag("-Wextra")
  #ei_add_cxx_compiler_flag("-Weverything")              # clang

  ei_add_cxx_compiler_flag("-Wundef")
  ei_add_cxx_compiler_flag("-Wcast-align")
  ei_add_cxx_compiler_flag("-Wchar-subscripts")
  ei_add_cxx_compiler_flag("-Wnon-virtual-dtor")
  ei_add_cxx_compiler_flag("-Wunused-local-typedefs")
  ei_add_cxx_compiler_flag("-Wpointer-arith")
  ei_add_cxx_compiler_flag("-Wwrite-strings")
  ei_add_cxx_compiler_flag("-Wformat-security")
  ei_add_cxx_compiler_flag("-Wshorten-64-to-32")
  ei_add_cxx_compiler_flag("-Wlogical-op")
  ei_add_cxx_compiler_flag("-Wenum-conversion")
  ei_add_cxx_compiler_flag("-Wc++11-extensions")
  ei_add_cxx_compiler_flag("-Wdouble-promotion")
#  ei_add_cxx_compiler_flag("-Wconversion")

  ei_add_cxx_compiler_flag("-Wshadow")

  ei_add_cxx_compiler_flag("-Wno-psabi")
  ei_add_cxx_compiler_flag("-Wno-variadic-macros")
  ei_add_cxx_compiler_flag("-Wno-long-long")

  ei_add_cxx_compiler_flag("-fno-check-new")
  ei_add_cxx_compiler_flag("-fno-common")
  ei_add_cxx_compiler_flag("-fstrict-aliasing")
  ei_add_cxx_compiler_flag("-wd981")                    # disable ICC's "operands are evaluated in unspecified order" remark
  ei_add_cxx_compiler_flag("-wd2304")                   # disable ICC's "warning #2304: non-explicit constructor with single argument may cause implicit type conversion" produced by -Wnon-virtual-dtor

  if(ANDROID_NDK)
    ei_add_cxx_compiler_flag("-pie")
    ei_add_cxx_compiler_flag("-fPIE")
  endif()

  set(CMAKE_REQUIRED_FLAGS "")

  option(EIGEN_TEST_SSE2 "Enable/Disable SSE2 in tests/examples" OFF)
  if(EIGEN_TEST_SSE2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse2")
    message(STATUS "Enabling SSE2 in tests/examples")
  endif()

  option(EIGEN_TEST_SSE3 "Enable/Disable SSE3 in tests/examples" OFF)
  if(EIGEN_TEST_SSE3)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse3")
    message(STATUS "Enabling SSE3 in tests/examples")
  endif()

  option(EIGEN_TEST_SSSE3 "Enable/Disable SSSE3 in tests/examples" OFF)
  if(EIGEN_TEST_SSSE3)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mssse3")
    message(STATUS "Enabling SSSE3 in tests/examples")
  endif()

  option(EIGEN_TEST_SSE4_1 "Enable/Disable SSE4.1 in tests/examples" OFF)
  if(EIGEN_TEST_SSE4_1)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.1")
    message(STATUS "Enabling SSE4.1 in tests/examples")
  endif()

  option(EIGEN_TEST_SSE4_2 "Enable/Disable SSE4.2 in tests/examples" OFF)
  if(EIGEN_TEST_SSE4_2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -msse4.2")
    message(STATUS "Enabling SSE4.2 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX "Enable/Disable AVX in tests/examples" OFF)
  if(EIGEN_TEST_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx")
    message(STATUS "Enabling AVX in tests/examples")
  endif()

  option(EIGEN_TEST_FMA "Enable/Disable FMA in tests/examples" OFF)
  if(EIGEN_TEST_FMA AND NOT EIGEN_TEST_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfma")
    message(STATUS "Enabling FMA in tests/examples")
  endif()

  option(EIGEN_TEST_AVX2 "Enable/Disable AVX2 in tests/examples" OFF)
  if(EIGEN_TEST_AVX2)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2 -mfma")
    message(STATUS "Enabling AVX2 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX512 "Enable/Disable AVX512 in tests/examples" OFF)
  if(EIGEN_TEST_AVX512)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512f -mfma")
    message(STATUS "Enabling AVX512 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX512DQ "Enable/Disable AVX512DQ in tests/examples" OFF)
  if(EIGEN_TEST_AVX512DQ)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx512dq -mfma")
    message(STATUS "Enabling AVX512DQ in tests/examples")
  endif()

  option(EIGEN_TEST_F16C "Enable/Disable F16C in tests/examples" OFF)
  if(EIGEN_TEST_F16C)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mf16c")
    message(STATUS "Enabling F16C in tests/examples")
  endif()

  option(EIGEN_TEST_ALTIVEC "Enable/Disable AltiVec in tests/examples" OFF)
  if(EIGEN_TEST_ALTIVEC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -maltivec -mabi=altivec")
    message(STATUS "Enabling AltiVec in tests/examples")
  endif()

  option(EIGEN_TEST_VSX "Enable/Disable VSX in tests/examples" OFF)
  if(EIGEN_TEST_VSX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m64 -mvsx")
    message(STATUS "Enabling VSX in tests/examples")
  endif()

  option(EIGEN_TEST_MSA "Enable/Disable MSA in tests/examples" OFF)
  if(EIGEN_TEST_MSA)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mmsa")
    message(STATUS "Enabling MSA in tests/examples")
  endif()

  option(EIGEN_TEST_NEON "Enable/Disable Neon in tests/examples" OFF)
  if(EIGEN_TEST_NEON)
    if(EIGEN_TEST_FMA)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon-vfpv4")
    else()
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpu=neon")
    endif()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfloat-abi=hard")
    message(STATUS "Enabling NEON in tests/examples")
  endif()

  option(EIGEN_TEST_NEON64 "Enable/Disable Neon in tests/examples" OFF)
  if(EIGEN_TEST_NEON64)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    message(STATUS "Enabling NEON in tests/examples")
  endif()

  option(EIGEN_TEST_Z13 "Enable/Disable S390X(zEC13) ZVECTOR in tests/examples" OFF)
  if(EIGEN_TEST_Z13)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=z13 -mzvector")
    message(STATUS "Enabling S390X(zEC13) ZVECTOR in tests/examples")
  endif()

  option(EIGEN_TEST_Z14 "Enable/Disable S390X(zEC14) ZVECTOR in tests/examples" OFF)
  if(EIGEN_TEST_Z14)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=z14 -mzvector")
    message(STATUS "Enabling S390X(zEC13) ZVECTOR in tests/examples")
  endif()

  check_cxx_compiler_flag("-fopenmp" COMPILER_SUPPORT_OPENMP)
  if(COMPILER_SUPPORT_OPENMP)
    option(EIGEN_TEST_OPENMP "Enable/Disable OpenMP in tests/examples" OFF)
    if(EIGEN_TEST_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
      message(STATUS "Enabling OpenMP in tests/examples")
    endif()
  endif()

else()

  # C4127 - conditional expression is constant
  # C4714 - marked as __forceinline not inlined (I failed to deactivate it selectively)
  #         We can disable this warning in the unit tests since it is clear that it occurs
  #         because we are oftentimes returning objects that have a destructor or may
  #         throw exceptions - in particular in the unit tests we are throwing extra many
  #         exceptions to cover indexing errors.
  # C4505 - unreferenced local function has been removed (impossible to deactivate selectively)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /EHsc /wd4127 /wd4505 /wd4714")

  # replace all /Wx by /W4
  string(REGEX REPLACE "/W[0-9]" "/W4" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

  check_cxx_compiler_flag("/openmp" COMPILER_SUPPORT_OPENMP)
  if(COMPILER_SUPPORT_OPENMP)
    option(EIGEN_TEST_OPENMP "Enable/Disable OpenMP in tests/examples" OFF)
    if(EIGEN_TEST_OPENMP)
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /openmp")
      message(STATUS "Enabling OpenMP in tests/examples")
    endif()
  endif()

  option(EIGEN_TEST_SSE2 "Enable/Disable SSE2 in tests/examples" OFF)
  if(EIGEN_TEST_SSE2)
    if(NOT CMAKE_CL_64)
      # arch is not supported on 64 bit systems, SSE is enabled automatically.
      set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:SSE2")
    endif()
    message(STATUS "Enabling SSE2 in tests/examples")
  endif()

  option(EIGEN_TEST_AVX "Enable/Disable AVX in tests/examples" OFF)
  if(EIGEN_TEST_AVX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX")
    message(STATUS "Enabling AVX in tests/examples")
  endif()

  option(EIGEN_TEST_FMA "Enable/Disable FMA/AVX2 in tests/examples" OFF)
  if(EIGEN_TEST_FMA AND NOT EIGEN_TEST_NEON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /arch:AVX2")
    message(STATUS "Enabling FMA/AVX2 in tests/examples")
  endif()

endif()

option(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION "Disable explicit vectorization in tests/examples" OFF)
option(EIGEN_TEST_X87 "Force using X87 instructions. Implies no vectorization." OFF)
option(EIGEN_TEST_32BIT "Force generating 32bit code." OFF)

if(EIGEN_TEST_X87)
  set(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION ON)
  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mfpmath=387")
    message(STATUS "Forcing use of x87 instructions in tests/examples")
  else()
    message(STATUS "EIGEN_TEST_X87 ignored on your compiler")
  endif()
endif()

if(EIGEN_TEST_32BIT)
  if(CMAKE_COMPILER_IS_GNUCXX)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -m32")
    message(STATUS "Forcing generation of 32-bit code in tests/examples")
  else()
    message(STATUS "EIGEN_TEST_32BIT ignored on your compiler")
  endif()
endif()

if(EIGEN_TEST_NO_EXPLICIT_VECTORIZATION)
  add_definitions(-DEIGEN_DONT_VECTORIZE=1)
  message(STATUS "Disabling vectorization in tests/examples")
endif()

option(EIGEN_TEST_NO_EXPLICIT_ALIGNMENT "Disable explicit alignment (hence vectorization) in tests/examples" OFF)
if(EIGEN_TEST_NO_EXPLICIT_ALIGNMENT)
  add_definitions(-DEIGEN_DONT_ALIGN=1)
  message(STATUS "Disabling alignment in tests/examples")
endif()

option(EIGEN_TEST_NO_EXCEPTIONS "Disables C++ exceptions" OFF)
if(EIGEN_TEST_NO_EXCEPTIONS)
  ei_add_cxx_compiler_flag("-fno-exceptions")
  message(STATUS "Disabling exceptions in tests/examples")
endif()

set(EIGEN_CUDA_CXX_FLAGS "" CACHE STRING "Additional flags to pass to the cuda compiler.")
set(EIGEN_CUDA_COMPUTE_ARCH 30 CACHE STRING "The CUDA compute architecture(s) to target when compiling CUDA code")

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Imported target support
add_library (eigen INTERFACE)
add_library (Eigen3::Eigen ALIAS eigen)
target_include_directories (eigen INTERFACE include)
