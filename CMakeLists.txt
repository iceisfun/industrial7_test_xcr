# Best-effort mirror of the Makefile for native Windows builds (NOT built in CI here; the Makefile with
# mingw-w64 is the reference). Supported compilers: clang-cl, clang (MSVC or GNU target), mingw gcc.
# Plain MSVC cl.exe is not supported: the suite needs GNU inline asm, top-level asm and target attributes.
#
#   cmake -S . -B build-cl -G Ninja -DCMAKE_C_COMPILER=clang-cl          (x64)
#   cmake -S . -B build-cl32 -G Ninja -DCMAKE_C_COMPILER=clang-cl -DCMAKE_C_FLAGS=-m32
#   cmake --build build-cl
#
# Never add /arch:AVX*, -mavx*, or -march: baseline exes must start on CPUs without AVX.
cmake_minimum_required(VERSION 3.16)
project(xcr_suite C)

if(MSVC AND NOT CMAKE_C_COMPILER_ID STREQUAL "Clang")
  message(FATAL_ERROR "use clang-cl (or mingw gcc): MSVC cl.exe lacks GNU inline asm / target attributes")
endif()

find_package(Git QUIET)
set(XCR_VERSION "unknown")
if(GIT_FOUND)
  execute_process(COMMAND ${GIT_EXECUTABLE} describe --always --dirty --tags
                  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR} OUTPUT_VARIABLE XCR_VERSION
                  OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
endif()

file(GLOB XCR_COMMON_SRC ${CMAKE_SOURCE_DIR}/common/*.c)
file(GLOB XCR_TESTS ${CMAKE_SOURCE_DIR}/tests/*.c)

add_library(xcr_common STATIC ${XCR_COMMON_SRC})
target_include_directories(xcr_common PUBLIC ${CMAKE_SOURCE_DIR}/common)
target_compile_definitions(xcr_common PUBLIC T_VERSION="${XCR_VERSION}" _CRT_SECURE_NO_WARNINGS)
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
  if(MSVC)
    target_compile_options(xcr_common PUBLIC /arch:SSE2)
  else()
    target_compile_options(xcr_common PUBLIC -msse2 -mstackrealign)
  endif()
endif()
if(MSVC)  # clang-cl
  target_compile_options(xcr_common PUBLIC /O2 -fno-strict-aliasing -Wno-unused-parameter)
else()
  target_compile_options(xcr_common PUBLIC -O2 -std=gnu11 -fno-strict-aliasing -fno-omit-frame-pointer)
endif()
target_link_libraries(xcr_common PUBLIC kernel32 advapi32 ntdll)

foreach(src ${XCR_TESTS})
  get_filename_component(name ${src} NAME_WE)
  add_executable(${name} ${src})
  # whole-archive is not needed: tests register through constructors in their own TU
  target_link_libraries(${name} PRIVATE xcr_common)
  if(NOT MSVC)
    target_link_options(${name} PRIVATE -static)
  endif()
  if(name STREQUAL "cet_xstate")   # CET shadow-stack compatible: same post-link step as the Makefile
    find_package(Python3 COMPONENTS Interpreter REQUIRED)
    add_custom_command(TARGET ${name} POST_BUILD
      COMMAND Python3::Interpreter ${CMAKE_SOURCE_DIR}/tools/pe_cetcompat.py $<TARGET_FILE:${name}>)
  endif()
endforeach()
