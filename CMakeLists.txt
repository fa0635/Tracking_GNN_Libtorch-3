cmake_minimum_required(VERSION 3.20)

#----------------------------------------------------------------------------------------------------------------------

project(TrackingGNN LANGUAGES CXX)

#----------------------------------------------------------------------------------------------------------------------

set(Torch_ROOT "/home/andrej/libtorch")
set(yaml-cpp_ROOT "/home/andrej/miniconda3")
set(ROOT_ROOT "/home/andrej/root_install")

find_package(Torch REQUIRED)
find_package(ROOT 6.36 CONFIG REQUIRED)
find_package(yaml-cpp REQUIRED)

#----------------------------------------------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS} -Wall -O2 -fPIC -ffast-math -mavx")

#----------------------------------------------------------------------------------------------------------------------

file(GLOB sources ${PROJECT_SOURCE_DIR}/src/*.cc)
file(GLOB train_source ${PROJECT_SOURCE_DIR}/train.cc)
file(GLOB evaluate_gnn_source ${PROJECT_SOURCE_DIR}/evaluate_gnn.cc)
file(GLOB evaluate_events_source ${PROJECT_SOURCE_DIR}/evaluate_events.cc)
file(GLOB data_preprocessing_source ${PROJECT_SOURCE_DIR}/data_preprocessing.cc)
file(GLOB train_data_preprocessing_source ${PROJECT_SOURCE_DIR}/train_data_preprocessing.cc)
file(GLOB train_data_preprocessing_2_source ${PROJECT_SOURCE_DIR}/train_data_preprocessing_2.cc)
file(GLOB train_2_source ${PROJECT_SOURCE_DIR}/train_2.cc)
file(GLOB evaluate_events_2_source ${PROJECT_SOURCE_DIR}/evaluate_events_2.cc)

#----------------------------------------------------------------------------------------------------------------------

separate_arguments(ROOT_EXE_LINKER_FLAGS)

#----------------------------------------------------------------------------------------------------------------------

add_library(TorchNetworks SHARED ${sources})
target_include_directories(TorchNetworks PUBLIC ${PROJECT_SOURCE_DIR}/include
                                                ${ROOT_INCLUDE_DIRS}
                                                ${YAML_CPP_INCLUDE_DIR})
target_compile_features(TorchNetworks PUBLIC cxx_std_17)
target_link_libraries(TorchNetworks PUBLIC ${TORCH_LIBRARIES}
                                           ${ROOT_LIBRARIES}
                                           ${ROOT_EXE_LINKER_FLAGS}
                                           ROOT::Geom
                                           yaml-cpp::yaml-cpp)
                                 
#----------------------------------------------------------------------------------------------------------------------

add_executable(train ${train_source})
target_include_directories(train PRIVATE ${PROJECT_SOURCE_DIR}/include
                                         ${ROOT_INCLUDE_DIRS})
target_compile_features(train PRIVATE cxx_std_17)
target_link_libraries(train PRIVATE TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(evaluate_gnn ${evaluate_gnn_source})
target_include_directories(evaluate_gnn PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                ${ROOT_INCLUDE_DIRS})
target_compile_features(evaluate_gnn PRIVATE cxx_std_17)
target_link_libraries(evaluate_gnn TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(evaluate_events ${evaluate_events_source})
target_include_directories(evaluate_events PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                   ${ROOT_INCLUDE_DIRS})
target_compile_features(evaluate_events PRIVATE cxx_std_17)
target_link_libraries(evaluate_events TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(data_preprocessing ${data_preprocessing_source})
target_include_directories(data_preprocessing PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                      ${ROOT_INCLUDE_DIRS})
target_compile_features(data_preprocessing PRIVATE cxx_std_17)
target_link_libraries(data_preprocessing TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(train_data_preprocessing ${train_data_preprocessing_source})
target_include_directories(train_data_preprocessing PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                            ${ROOT_INCLUDE_DIRS})
target_compile_features(train_data_preprocessing PRIVATE cxx_std_17)
target_link_libraries(train_data_preprocessing TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(train_data_preprocessing_2 ${train_data_preprocessing_2_source})
target_include_directories(train_data_preprocessing_2 PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                            ${ROOT_INCLUDE_DIRS})
target_compile_features(train_data_preprocessing_2 PRIVATE cxx_std_17)
target_link_libraries(train_data_preprocessing_2 TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(train_2 ${train_2_source})
target_include_directories(train_2 PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                            ${ROOT_INCLUDE_DIRS})
target_compile_features(train_2 PRIVATE cxx_std_17)
target_link_libraries(train_2 TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------

add_executable(evaluate_events_2 ${evaluate_events_2_source})
target_include_directories(evaluate_events_2 PRIVATE ${PROJECT_SOURCE_DIR}/include
                                                            ${ROOT_INCLUDE_DIRS})
target_compile_features(evaluate_events_2 PRIVATE cxx_std_17)
target_link_libraries(evaluate_events_2 TorchNetworks)

#----------------------------------------------------------------------------------------------------------------------
