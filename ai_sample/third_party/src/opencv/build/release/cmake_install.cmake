# Install script for directory: /home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv

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
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/licenses/opencv4" TYPE FILE RENAME "ade-LICENSE" FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/3rdparty/ade/ade-0.1.1f/LICENSE")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/cvconfig.h")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/include/opencv4/opencv2" TYPE FILE FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/opencv2/opencv_modules.hpp")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4/OpenCVModules.cmake")
    file(DIFFERENT EXPORT_FILE_CHANGED FILES
         "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4/OpenCVModules.cmake"
         "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/CMakeFiles/Export/lib/cmake/opencv4/OpenCVModules.cmake")
    if(EXPORT_FILE_CHANGED)
      file(GLOB OLD_CONFIG_FILES "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4/OpenCVModules-*.cmake")
      if(OLD_CONFIG_FILES)
        message(STATUS "Old export file \"$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4/OpenCVModules.cmake\" will be replaced.  Removing files [${OLD_CONFIG_FILES}].")
        file(REMOVE ${OLD_CONFIG_FILES})
      endif()
    endif()
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4" TYPE FILE FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/CMakeFiles/Export/lib/cmake/opencv4/OpenCVModules.cmake")
  if("${CMAKE_INSTALL_CONFIG_NAME}" MATCHES "^([Rr][Ee][Ll][Ee][Aa][Ss][Ee])$")
    file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4" TYPE FILE FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/CMakeFiles/Export/lib/cmake/opencv4/OpenCVModules-release.cmake")
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/cmake/opencv4" TYPE FILE FILES
    "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/unix-install/OpenCVConfig-version.cmake"
    "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/unix-install/OpenCVConfig.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xscriptsx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/bin" TYPE FILE PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE FILES "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/CMakeFiles/install/setup_vars_opencv4.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xdevx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/opencv4" TYPE FILE FILES
    "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/platforms/scripts/valgrind.supp"
    "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/platforms/scripts/valgrind_3rdparty.supp"
    )
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/3rdparty/zlib/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/3rdparty/libjpeg-turbo/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/3rdparty/openjpeg/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/3rdparty/libpng/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/3rdparty/protobuf/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/include/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/calib3d/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/core/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/dnn/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/features2d/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/flann/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/gapi/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/highgui/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/imgcodecs/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/imgproc/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/java/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/js/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/ml/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/objc/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/objdetect/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/photo/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/python/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/stitching/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/ts/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/video/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/videoio/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/.firstpass/world/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/python_tests/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/world/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/java_bindings_generator/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/js_bindings_generator/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/objc_bindings_generator/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/modules/python_bindings_generator/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/doc/cmake_install.cmake")
  include("/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/data/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/openharmony/sdk_linux/sample/taurus/ai_sample/third_party/src/opencv/build/release/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
