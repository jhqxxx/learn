CMake是一个编译工具，允许开发者编写一种平台无关的CmakeLists.txt文件来定制整个编译流程
##### 常见指令
* 指定CMake最低版本要求：
    - cmake_minimum_required(VERSION <version>)
    - 示例：cmake_minimum_required(VERSION 3.5)
* 定义项目的名称和使用的编程语言：
    - project(<project_name> [<language>...])
    - 示例：project(MyProject CXX)
    - 可定义项目版本：project(MyProject VERSION 1.0)
* 指定要生成的可执行文件和其源文件：
    - add_executable(<target> <source_files>...)
    - 示例：add_executable(MyExecutable main.cpp other_file.cpp)
* 创建一个库（静态库或动态库）及其源文件：
    - add_library(<target> <type> <source_files>...)
    - 示例：add_library(MyLibrary STATIC library.cpp other_library.cpp)
    - 可选类型：STATIC（静态库）、SHARED（动态库）
        - 静态库：在编译时整合到可执行文件，增加文件大小，但不需要运行时依赖，适合小规模，不频繁更新的场景。
        - 动态库：在运行时加载，减小文件大小，允许多个程序共享，便于更新，适合大型，多程序共享或需要动态更新的情况，动态库利用延迟加载技术优化内存使用
* 链接目标文件与其他库：
    - target_link_libraries(<target> <libraries>...)
    - 示例：target_link_libraries(MyExecutable MyLibrary)
* 添加头文件搜索路径
    - include_directories(<dirs>...)
    - 示例：include_directories<${PROJECT_SOURCE_DIR}/include>
* 设置变量的值
    - set(<variable> <value>)
    - 示例：set(CMAKE_CXX_STANDARD 11)    
* 为特定目标指定头文件目录：
    - target_include_directories(<target> <PUBLIC|PRIVATE|INTERFACE> <dirs>...)
    - 示例：target_include_directories(MyExecutable PUBLIC ${PROJECT_SOURCE_DIR}/include)
* 安装规则：
    - install(TARGETS <targets>... [RUNTIME DESTINATION dir]         
                                   [LIBRARY DESTINATION dir]         
                                   [ARCHIVE DESTINATION dir]         
                                   [INCLUDES DESTINATION [dir ...]]         [PRIVATE_HEADER DESTINATION dir]         [PUBLIC_HEADER DESTINATION dir])
    - 示例：install(TARGETS MyExecutable RUNTIME DESTINATION bin)
* 条件语句(if, elseif, else, endif):
    - if(<condition>)
        <commands>
      elseif(<condition>)
        <commands>
      else()
        <commands>
      endif()
    - 示例：if(CMAKE_BUILD_TYPE STREQUAL "Debug")
                message("Debug build)
            endif()
* 自定义命令：
    - add_custom_command(
        TARGET target    
        PRE_BUILD | PRE_LINK | POST_BUILD    
        COMMAND command1 [ARGS] [WORKING_DIRECTORY dir]    
        [COMMAND command2 [ARGS]]    
        [DEPENDS [depend1 [depend2 ...]]]    
        [COMMENT comment]    
        [VERBATIM])
    - 示例：add_custom_command(
            TARGET MyExecutable POST_BUILD 
            COMMAND ${CMAKE_COMMAND} -E echo "Build completed.")
* 查找库和包：使用find_package()指令自动检测和配置外部库和包，常用于查找系统安装的库或第三方库
    - find_package(<package> [version] [EXACT] [QUIET] [MODULE] [REQUIRED])
    - 示例：find_package(Boost REQUIRED)
    - 设置查找路径：set(Boost_ROOT "/path/to/boost")
    - 指定版本：find_package(Boost 1.67.0 REQUIRED)
    - 查找库并指定路径:find_package(Boost 1.67.0 REQUIRED PATHS /path/to/boost)
    - 使用查找到的库：target_link_libraries(MyExecutable Boost::boost)
    - 设置包含目录：include_directories(${Boost_INCLUDE_DIRS})
    - 链接目标：link_directories(${Boost_LIBRARY_DIRS})
    - 实际上find_package()是在寻找包的配置文件，Boost为例，是在找名为BoostConfig.cmake的配置文件，一般在安装库的时候，会带一个NAME.cmake文件在系统的cmake文件里。
* 查找库的另一种方式：find_library,它是一个更加基础的方法，它不依赖库提供的cmake配置文件，而是直接查找，需要作者手动指定文件路径
* 获取某目录下的所有源文件：
    - aux_source_directory(<dirs> <variable>)
* 添加编译选项：
    - add_compile_options()
* file():
    - file(GLOB_RECURSE utils_src CONFIGURE_DEPENDS utils/*.cpp)：递归搜索指定目录下的所有源文件，并将结果保存到变量utils_src中
* add_subdirectory(<dirs>)：将子目录添加到构建
* option(<option_variable> "help string describing option" [initial value])
    - 给用户提供编译选项，如果没有提供，则使用初始值
    - cmake -D<option_variable>=<value> ..
* add_definitions()：用于在源文件的编译中添加-D定义标志
* configure_file(<input> <output>                
                [COPYONLY] [ESCAPE_QUOTES] [@ONLY]                [NEWLINE_STYLE [UNIX|DOS|WIN32|LF|CRLF] ])
    - 将输入文件复制到输出文件，并根据需要替换变量和宏
##### 构建流程
* 创建构建目录：mkdir build
* 进入构建目录：cd build
* 使用cmake生成构建文件： cmake ..
    - 如果需要指定生成器(如Ninja, Visual Studio等),使用-G选项：cmake -G "Ninja" ..
    - 如果需要指定构建类型(如Debug, Release等), 使用-DCMAKE_BUILD_TYPE选项：cmake -DCMAKE_BUILD_TYPE=Debug ..
* 编译和构建
    - 使用Makefile构建：make
    - 使用Ninja构建：ninja
* 清理构建文件
    - 使用Makefile清理: make clean
    - 使用Ninja清理: ninja clean
    - 手动删除： rm -rf build/*
* 重新配置和构建：修改CMakeLists.txt文件后，重新执行cmake命令重新生成构建文件，然后执行make命令重新构建项目。

##### 变量说明
* ${PROJECT_BINARY_DIR}:是cmake系统变量，执行cmake的目录，如果在build目录下，则这个变量指build目录

##### 交叉编译
export GCC_COMPILER=/home/wxg/jhq/code/about_rknn/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu
export CC=${GCC_COMPILER}-gcc
export CXX=${GCC_COMPILER}-g++

cmake .. -DTARGET_SOC=rk3588 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_BUILD_TYPE=Release -DENABLE_ASAN=OFF -DCMAKE_C_COMPILER=/home/wxg/jhq/code/about_rknn/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc -DCMAKE_CXX_COMPILER=/home/wxg/jhq/code/about_rknn/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ -DCMAKE_INSTALL_PREFIX=../lib

opencv4:
cmake -D TARGET_SOC=rk3588 -D CMAKE_SYSTEM_NAME=Linux -D CMAKE_SYSTEM_PROCESSOR=aarch64 -D CMAKE_BUILD_TYPE=Release -D CMAKE_BUILD_TYPE=RELEASE -D CMAKE_C_COMPILER=/home/wxg/jhq/code/about_rknn/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc -D CMAKE_CXX_COMPILER=/home/wxg/jhq/code/about_rknn/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ -D BUILD_SHARED_LIBS=OFF -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC -D CMAKE_EXE_LINKER_FLAGS="-lpthread -ldl" -D ENABLE_PIC=OFF -D WITH_1394=OFF -D WITH_ARAVIS=OFF -D WITH_ARITH_DEC=ON -D WITH_ARITH_ENC=ON -D WITH_CLP=OFF  -D WITH_CUBLAS=OFF  -D WITH_CUDA=OFF  -D WITH_CUFFT=OFF  -D WITH_FFMPEG=OFF  -D WITH_GSTREAMER=ON  -D WITH_GSTREAMER_0_10=OFF  -D WITH_HALIDE=OFF  -D WITH_HPX=OFF  -D WITH_IMGCODEC_HDR=ON  -D WITH_IMGCODEC_PXM=ON  -D WITH_IMGCODEC_SUNRASTER=ON  -D WITH_INF_ENGINE=OFF  -D WITH_IPP=OFF  -D WITH_ITT=OFF  -D WITH_JASPER=ON  -D WITH_JPEG=ON  -D WITH_LAPACK=ON  -D WITH_LIBREALSENSE=OFF  -D WITH_NVCUVID=OFF  -D WITH_OPENCL=OFF  -D WITH_OPENCLAMDBLAS=OFF  -D WITH_OPENCLAMDFFT=OFF  -D WITH_OPENCL_SVM=OFF  -D WITH_OPENEXR=OFF  -D WITH_OPENGL=OFF  -D WITH_OPENMP=OFF  -D WITH_OPENNNI=OFF  -D WITH_OPENNNI2=OFF  -D WITH_OPENVX=OFF  -D WITH_PNG=ON  -D WITH_PROTOBUF=OFF  -D WITH_PTHREADS_PF=ON  -D WITH_PVAPI=OFF  -D WITH_QT=OFF  -D WITH_QUIRC=OFF   -D WITH_TBB=OFF  -D WITH_TIFF=ON  -D WITH_VULKAN=OFF  -D WITH_WEBP=ON  -D WITH_XIMEA=OFF  -D BUILD_JAVA=OFF  -D BUILD_opencv_python3=OFF  -D CMAKE_INSTALL_PREFIX=../../install/aarch64   -D WITH_GTK=OFF  -D WITH_EIGEN=OFF  -D OPENCV_EXTRA_MODULES_PATH="../../../opencv_contrib/modules"  -D BUILD_ZLIB=ON  -D ZLIB_INCLUDE_DIR="../../3rdparty/zlib"  -D OPENCV_GENERATE_PKGCONFIG=ON  ../..


cmake -D CMAKE_SYSTEM_NAME=Linux -D CMAKE_SYSTEM_PROCESSOR=x64  -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=OFF -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC -D CMAKE_EXE_LINKER_FLAGS="-lpthread -ldl" -D ENABLE_PIC=OFF -D WITH_1394=OFF -D WITH_ARAVIS=OFF -D WITH_ARITH_DEC=ON -D WITH_ARITH_ENC=ON -D WITH_CLP=OFF  -D WITH_CUBLAS=OFF  -D WITH_CUDA=OFF  -D WITH_CUFFT=OFF  -D WITH_FFMPEG=OFF  -D WITH_GSTREAMER=ON  -D WITH_GSTREAMER_0_10=OFF  -D WITH_HALIDE=OFF  -D WITH_HPX=OFF  -D WITH_IMGCODEC_HDR=ON  -D WITH_IMGCODEC_PXM=ON  -D WITH_IMGCODEC_SUNRASTER=ON  -D WITH_INF_ENGINE=OFF  -D WITH_IPP=OFF  -D WITH_ITT=OFF  -D WITH_JASPER=ON  -D WITH_JPEG=ON  -D WITH_LAPACK=ON  -D WITH_LIBREALSENSE=OFF  -D WITH_NVCUVID=OFF  -D WITH_OPENCL=OFF  -D WITH_OPENCLAMDBLAS=OFF  -D WITH_OPENCLAMDFFT=OFF  -D WITH_OPENCL_SVM=OFF  -D WITH_OPENEXR=OFF  -D WITH_OPENGL=OFF  -D WITH_OPENMP=OFF  -D WITH_OPENNNI=OFF  -D WITH_OPENNNI2=OFF  -D WITH_OPENVX=OFF  -D WITH_PNG=ON  -D WITH_PROTOBUF=ON  -D WITH_PTHREADS_PF=ON  -D WITH_PVAPI=OFF  -D WITH_QT=OFF  -D WITH_QUIRC=OFF   -D WITH_TBB=OFF  -D WITH_TIFF=ON  -D WITH_VULKAN=OFF  -D WITH_WEBP=ON  -D WITH_XIMEA=OFF  -D BUILD_JAVA=OFF  -D BUILD_opencv_python3=OFF  -D CMAKE_INSTALL_PREFIX=../../install/x64   -D WITH_GTK=OFF  -D WITH_EIGEN=OFF -D BUILD_ZLIB=ON -D OPENCV_EXTRA_MODULES_PATH="../../../opencv_contrib/modules" -D ZLIB_INCLUDE_DIR="../../3rdparty/zlib"  -D OPENCV_GENERATE_PKGCONFIG=ON  ../..

opencv-wsl:
cmake -D CMAKE_SYSTEM_NAME=Linux -D CMAKE_SYSTEM_PROCESSOR=x64  -D CMAKE_BUILD_TYPE=Release -D BUILD_SHARED_LIBS=OFF -D CMAKE_CXX_FLAGS=-fPIC -D CMAKE_C_FLAGS=-fPIC -D CMAKE_EXE_LINKER_FLAGS="-lpthread -ldl" -D ENABLE_PIC=OFF -D WITH_1394=OFF -D WITH_ARAVIS=OFF -D WITH_ARITH_DEC=ON -D WITH_ARITH_ENC=ON -D WITH_CLP=OFF  -D WITH_CUBLAS=OFF  -D WITH_CUDA=OFF  -D WITH_CUFFT=OFF  -D WITH_FFMPEG=OFF  -D WITH_GSTREAMER=ON  -D WITH_GSTREAMER_0_10=OFF  -D WITH_HALIDE=OFF  -D WITH_HPX=OFF  -D WITH_IMGCODEC_HDR=ON  -D WITH_IMGCODEC_PXM=ON  -D WITH_IMGCODEC_SUNRASTER=ON  -D WITH_INF_ENGINE=OFF  -D WITH_IPP=OFF  -D WITH_ITT=OFF  -D WITH_JASPER=ON  -D WITH_JPEG=ON  -D WITH_LAPACK=ON  -D WITH_LIBREALSENSE=OFF  -D WITH_NVCUVID=OFF  -D WITH_OPENCL=OFF  -D WITH_OPENCLAMDBLAS=OFF  -D WITH_OPENCLAMDFFT=OFF  -D WITH_OPENCL_SVM=OFF  -D WITH_OPENEXR=OFF  -D WITH_OPENGL=OFF  -D WITH_OPENMP=OFF  -D WITH_OPENNNI=OFF  -D WITH_OPENNNI2=OFF  -D WITH_OPENVX=OFF  -D WITH_PNG=ON  -D WITH_PROTOBUF=ON  -D WITH_PTHREADS_PF=ON  -D WITH_PVAPI=OFF  -D WITH_QT=OFF  -D WITH_QUIRC=OFF   -D WITH_TBB=OFF  -D WITH_TIFF=ON  -D WITH_VULKAN=OFF  -D WITH_WEBP=ON  -D WITH_XIMEA=OFF  -D BUILD_JAVA=OFF  -D BUILD_opencv_python3=OFF  -D CMAKE_INSTALL_PREFIX=../../install/x64   -D WITH_GTK=OFF  -D WITH_EIGEN=OFF -D BUILD_OPENJPEG=ON -D BUILD_JASPER=ON -D BUILD_OPENJPEG2=ON -D BUILD_ZLIB=ON -D OPENCV_EXTRA_MODULES_PATH="../../../opencv_contrib/modules" -D ZLIB_INCLUDE_DIR="../../3rdparty/zlib"  -D OPENCV_GENERATE_PKGCONFIG=ON   ../..

cmake .. -DTARGET_SOC=rk3588 -DCMAKE_SYSTEM_NAME=Linux -DCMAKE_SYSTEM_PROCESSOR=aarch64 -DCMAKE_BUILD_TYPE=Release -D CMAKE_C_COMPILER=/home/wxg/jhq/code/about_rknn/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-gcc -D CMAKE_CXX_COMPILER=/home/wxg/jhq/code/about_rknn/gcc-linaro-6.3.1-2017.05-x86_64_aarch64-linux-gnu/bin/aarch64-linux-gnu-g++ -DCMAKE_INSTALL_PREFIX=../demo