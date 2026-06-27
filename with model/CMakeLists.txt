cmake_minimum_required(VERSION 3.18)

project(search_engine)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")

find_package(Torch REQUIRED)

add_library(search_engine SHARED
    search_engine.cpp
)

target_link_libraries(search_engine
    "${TORCH_LIBRARIES}"
)

set_property(TARGET search_engine PROPERTY CXX_STANDARD 17)
