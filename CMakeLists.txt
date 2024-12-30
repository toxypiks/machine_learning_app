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

set(ADDER_GEN_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/adder_gen.c
  )

add_executable(adder_gen)
target_sources(adder_gen
 PRIVATE
  ${ADDER_GEN_SRC}
  )

target_link_libraries(adder_gen
  ${raylib_LIBRARIES}
  m
  )

set(XOR_GEN_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/xor_gen.c
  )

add_executable(xor_gen)
target_sources(xor_gen
 PRIVATE
  ${XOR_GEN_SRC}
  )

target_link_libraries(xor_gen
  ${raylib_LIBRARIES}
  m
  )

set(PNG2MAT_SRC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/png2mat.c
  )

add_executable(png2mat)
target_sources(png2mat
 PRIVATE
  ${PNG2MAT_SRC}
  )

target_link_libraries(png2mat
  ${raylib_LIBRARIES}
  m
  )
