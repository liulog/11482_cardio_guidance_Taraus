# Install script for directory: /home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/world

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/install")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "TRUE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlicensesx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/licenses/opencv4" TYPE FILE RENAME "SoftFloat-COPYING.txt" FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/3rdparty/SoftFloat/COPYING.txt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xlibsx" OR NOT CMAKE_INSTALL_COMPONENT)
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so.4.5.5"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so.405"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHECK
           FILE "${file}"
           RPATH "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/install/lib")
    endif()
  endforeach()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY OPTIONAL FILES
    "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/lib/libopencv_world.so.4.5.5"
    "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/lib/libopencv_world.so.405"
    )
  foreach(file
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so.4.5.5"
      "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so.405"
      )
    if(EXISTS "${file}" AND
       NOT IS_SYMLINK "${file}")
      file(RPATH_CHANGE
           FILE "${file}"
           OLD_RPATH "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
           NEW_RPATH "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/install/lib")
      if(CMAKE_INSTALL_DO_STRIP)
        execute_process(COMMAND "/home/openharmony/prebuilts/clang/ohos/linux-x86_64/llvm/bin/llvm-strip" "${file}")
      endif()
    endif()
  endforeach()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so"
         RPATH "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/install/lib")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib" TYPE SHARED_LIBRARY FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/lib/libopencv_world.so")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so"
         OLD_RPATH "::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::"
         NEW_RPATH "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/install/lib")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/home/openharmony/prebuilts/clang/ohos/linux-x86_64/llvm/bin/llvm-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/libopencv_world.so")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/calib3d/include/opencv2/calib3d.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/calib3d" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/calib3d/include/opencv2/calib3d/calib3d.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/calib3d" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/calib3d/include/opencv2/calib3d/calib3d_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/affine.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/async.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/base.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/bindings_utils.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/bufferpool.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/check.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/core_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda.inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/block.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/border_interpolate.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/color.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/common.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/datamov_utils.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/detail/color_detail.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/detail/reduce.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/detail/reduce_key_val.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/detail/transform_detail.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/detail/type_traits_detail.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/detail/vec_distance_detail.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/dynamic_smem.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/emulation.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/filters.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/funcattrib.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/functional.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/limits.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/reduce.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/saturate_cast.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/scan.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/simd_functions.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/transform.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/type_traits.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/utility.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/vec_distance.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/vec_math.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/vec_traits.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/warp.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/warp_reduce.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/cuda" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda/warp_shuffle.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda_stream_accessor.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cuda_types.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cv_cpu_dispatch.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cv_cpu_helper.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cvdef.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cvstd.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cvstd.inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/cvstd_wrapper.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/detail/async_promise.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/detail/dispatch_helper.impl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/detail/exception_ptr.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/directx.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/dualquaternion.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/dualquaternion.inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/eigen.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/fast_math.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/hal.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/interface.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_avx.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_avx512.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_cpp.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_forward.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_msa.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_neon.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_rvv.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_rvv071.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_sse.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_sse_em.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_vsx.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/intrin_wasm.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/msa_macros.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/hal/simd_utils.impl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/mat.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/mat.inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/matx.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/neon_utils.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/ocl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/ocl_genbase.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/ocl_defs.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/opencl_info.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/opencl_svm.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime/autogenerated" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/autogenerated/opencl_clblas.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime/autogenerated" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/autogenerated/opencl_clfft.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime/autogenerated" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/autogenerated/opencl_core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime/autogenerated" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/autogenerated/opencl_core_wrappers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime/autogenerated" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/autogenerated/opencl_gl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime/autogenerated" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/autogenerated/opencl_gl_wrappers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_clblas.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_clfft.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_core_wrappers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_gl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_gl_wrappers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_svm_20.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_svm_definitions.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/opencl/runtime" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opencl/runtime/opencl_svm_hsa_extension.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/opengl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/operations.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/optim.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/ovx.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/parallel/backend" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/parallel/backend/parallel_for.openmp.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/parallel/backend" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/parallel/backend/parallel_for.tbb.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/parallel" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/parallel/parallel_backend.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/persistence.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/quaternion.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/quaternion.inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/saturate.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/simd_intrinsics.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/softfloat.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/sse_utils.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/traits.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/types.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/types_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utility.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/allocator_stats.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/allocator_stats.impl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/filesystem.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/instrumentation.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/logger.defines.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/logger.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/logtag.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/tls.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/utils/trace.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/va_intel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/version.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/core" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/core/include/opencv2/core/vsx_utils.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/all_layers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/dict.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/dnn.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/dnn.inl.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/layer.details.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/layer.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/shape_utils.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/utils/debug_utils.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn/utils" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/utils/inference_engine.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/dnn" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/dnn/include/opencv2/dnn/version.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/features2d/include/opencv2/features2d.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/features2d" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/features2d/include/opencv2/features2d/features2d.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/features2d/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/features2d/include/opencv2/features2d/hal/interface.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/all_indices.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/allocator.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/any.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/autotuned_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/composite_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/config.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/defines.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/dist.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/dummy.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/dynamic_bitset.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/flann.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/flann_base.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/general.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/ground_truth.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/hdf5.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/heap.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/hierarchical_clustering_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/index_testing.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/kdtree_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/kdtree_single_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/kmeans_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/linear_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/logger.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/lsh_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/lsh_table.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/matrix.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/miniflann.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/nn_index.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/object_factory.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/params.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/random.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/result_set.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/sampling.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/saving.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/simplex_downhill.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/flann" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/flann/include/opencv2/flann/timer.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/cpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/cpu/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/cpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/cpu/gcpukernel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/cpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/cpu/imgproc.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/cpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/cpu/stereo.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/cpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/cpu/video.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/fluid" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/fluid/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/fluid" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/fluid/gfluidbuffer.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/fluid" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/fluid/gfluidkernel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/fluid" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/fluid/imgproc.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/garg.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/garray.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gasync_context.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gcall.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gcommon.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gcompiled.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gcompiled_async.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gcompoundkernel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gcomputation.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gcomputation_async.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gframe.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gkernel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gmat.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gmetaarg.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gopaque.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gproto.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/gpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gpu/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/gpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gpu/ggpukernel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/gpu" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gpu/imgproc.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gscalar.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gstreaming.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gtransform.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gtype_traits.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/gtyped.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/imgproc.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/infer.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/infer" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/infer/bindings_ie.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/infer" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/infer/ie.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/infer" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/infer/onnx.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/infer" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/infer/parsers.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/media.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/ocl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/ocl/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/ocl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/ocl/goclkernel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/ocl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/ocl/imgproc.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/opencv_includes.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/operators.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/assert.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/convert.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/cvdefs.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/exports.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/mat.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/saturate.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/scalar.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/own" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/own/types.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/plaidml" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/plaidml/core.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/plaidml" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/plaidml/gplaidmlkernel.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/plaidml" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/plaidml/plaidml.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/python" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/python/python.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/render.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/render" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/render/render.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/render" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/render/render_types.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/rmat.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/s11n.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/s11n" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/s11n/base.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/stereo.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/cap.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/desync.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/format.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming/gstreamer" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/gstreamer/gstreamerpipeline.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming/gstreamer" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/gstreamer/gstreamersource.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/meta.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming/onevpl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/onevpl/cfg_params.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming/onevpl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/onevpl/data_provider_interface.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming/onevpl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/onevpl/device_selector_interface.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming/onevpl" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/onevpl/source.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/source.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/streaming" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/streaming/sync.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/any.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/compiler_hints.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/copy_through_move.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/optional.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/throw.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/type_traits.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/util.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi/util" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/util/variant.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/gapi" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/gapi/include/opencv2/gapi/video.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/highgui/include/opencv2/highgui.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/highgui" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/highgui/include/opencv2/highgui/highgui.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/highgui" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/highgui/include/opencv2/highgui/highgui_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgcodecs/include/opencv2/imgcodecs.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgcodecs" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgcodecs/include/opencv2/imgcodecs/imgcodecs.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgcodecs" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgcodecs/include/opencv2/imgcodecs/imgcodecs_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgcodecs" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgcodecs/include/opencv2/imgcodecs/ios.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgcodecs/legacy" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgcodecs/include/opencv2/imgcodecs/legacy/constants_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgcodecs" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgcodecs/include/opencv2/imgcodecs/macosx.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/bindings.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc/detail" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/detail/gcgraph.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/hal/hal.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc/hal" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/hal/interface.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/imgproc.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/imgproc_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/segmentation.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/imgproc" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/imgproc/include/opencv2/imgproc/types_c.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/objdetect/include/opencv2/objdetect.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/objdetect" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/objdetect/include/opencv2/objdetect/detection_based_tracker.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/objdetect" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/objdetect/include/opencv2/objdetect/face.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2/objdetect" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/objdetect/include/opencv2/objdetect/objdetect.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE OPTIONAL FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/modules/world/include/opencv2/world.hpp")
endif()

