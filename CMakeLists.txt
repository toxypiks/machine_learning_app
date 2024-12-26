cmake_minimum_required(VERSION 3.16)
project(machine_learning)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(include)

find_package(raylib)

set(GYM_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/gym.c
  )

add_executable(gym_src)
target_sources(gym_src
 PRIVATE
  ${GYM_SRC}
  )

target_include_directories(gym_src
 PRIVATE
  ${raylib_INCLUDE_DIRS}
  )

target_link_libraries(gym_src
  ${raylib_LIBRARIES}
  m
  )
