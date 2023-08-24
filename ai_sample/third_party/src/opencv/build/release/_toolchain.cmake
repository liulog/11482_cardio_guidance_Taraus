# Specify target environment
set(CMAKE_SYSTEM_NAME Linux)
# Specify LLVM as cross-compilation toolchain
set(LLVM_ROOT "/home/openharmony/prebuilts/clang/ohos/linux-x86_64/llvm")
set(CLANG_TARGET_TRIPLE arm-linux-ohos)
set(CMAKE_C_COMPILER ${LLVM_ROOT}/bin/clang)
set(CMAKE_C_COMPILER_TARGET ${CLANG_TARGET_TRIPLE})
set(CMAKE_CXX_COMPILER ${LLVM_ROOT}/bin/clang++)
set(CMAKE_CXX_COMPILER_TARGET ${CLANG_TARGET_TRIPLE})
set(CMAKE_ASM_COMPILER ${LLVM_ROOT}/bin/clang)
set(CMAKE_ASM_COMPILER_TARGET ${CLANG_TARGET_TRIPLE})
set(CMAKE_AR ${LLVM_ROOT}/bin/llvm-ar)
set(CMAKE_LINKER ${LLVM_ROOT}/bin/ld.lld)
set(CMAKE_NM      ${LLVM_ROOT}/bin/llvm-nm)
set(CMAKE_OBJDUMP ${LLVM_ROOT}/bin/llvm-objdump)
set(CMAKE_RANLIB  ${LLVM_ROOT}/bin/llvm-ranlib)
set(LLVM_CONFIG_PATH ${LLVM_ROOT}/bin/llvm-config)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
# Use LLVM's lld linker
set(CMAKE_EXE_LINKER_FLAGS "-fuse-ld=lld")
set(CMAKE_SHARED_LINKER_FLAGS "-fuse-ld=lld")
# Don't run the linker on compiler check
#set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
# Specify sysroot of target platform
set(SYSROOT "/home/openharmony/out/hispark_taurus/ipcamera_hispark_taurus_linux/sysroot")
set(CMAKE_C_COMPILER_EXTERNAL_TOOLCHAIN "/home/openharmony/prebuilts/gcc/linux-x86/arm/gcc-linaro-7.5.0-arm-linux-gnueabi/bin/arm-linux-gnueabi-gcc")
set(CMAKE_CXX_COMPILER_EXTERNAL_TOOLCHAIN "/home/openharmony/prebuilts/gcc/linux-x86/arm/gcc-linaro-7.5.0-arm-linux-gnueabi/bin/arm-linux-gnueabi-g++")
set(CMAKE_SYSROOT ${SYSROOT})

# Specify cmake search path
set(CMAKE_FIND_ROOT_PATH ${SYSROOT})
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)

# set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -L/home/lvhaifeng/ohos/l1code/prebuilts/gcc/linux-x86/arm/gcc-linaro-7.5.0-arm-linux-gnueabi/arm-linux-gnueabi/lib -latomic")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=armv7-a")
