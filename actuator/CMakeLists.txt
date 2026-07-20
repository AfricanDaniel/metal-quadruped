cmake_minimum_required(VERSION 3.20)
project(actuator)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(yaml-cpp REQUIRED)

# Older yaml-cpp CMake configs (e.g. the 0.6.x shipped with Ubuntu 18.04 /
# JetPack on Jetson boards) only export a plain "yaml-cpp" target instead of
# the namespaced "yaml-cpp::yaml-cpp" alias used by newer versions.
if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp)
else()
  set(YAML_CPP_TARGET ${YAML_CPP_LIBRARIES})
endif()

rosidl_generate_interfaces(${PROJECT_NAME}
  "srv/AdjustMotorPosition.srv"
  "srv/SetMotorVelocity.srv"
  "srv/ReadMotorPositions.srv"
  "srv/GoToPose.srv"
  "srv/SetHome.srv"
)

# Unitree SDK — sits two levels above the ros2 workspace root
set(SDK_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../unitree_actuator_sdk)
include_directories(${SDK_DIR}/include)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    set(UNITREE_LIB ${SDK_DIR}/lib/libUnitreeMotorSDK_Arm64.so)
else()
    set(UNITREE_LIB ${SDK_DIR}/lib/libUnitreeMotorSDK_Linux64.so)
endif()

# ROS 2 node: exposes services (set_motor_velocity, adjust_motor_position,
# read_motor_positions, go_to_pose) to drive/read all motors on the bus.
add_executable(basic_control src/basic_control.cpp)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} rosidl_typesupport_cpp)
target_link_libraries(basic_control ${UNITREE_LIB} ${YAML_CPP_TARGET} "${cpp_typesupport_target}")
# Bake in the package's source directory so data/ and config/ always resolve
# to actuator/data and actuator/config regardless of where `ros2 run` is
# invoked from.
target_compile_definitions(basic_control PRIVATE
  ACTUATOR_PACKAGE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")
ament_target_dependencies(basic_control rclcpp)
set_target_properties(basic_control PROPERTIES
    INSTALL_RPATH "${SDK_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

# Plain C++ CLI test tool: no ROS 2 involved at runtime (no rclcpp link), just
# the Unitree SDK. Good for quick bring-up testing of a single motor by ID
# without needing a ROS 2 node/services running.
add_executable(motor_cli src/motor_cli.cpp)
target_link_libraries(motor_cli ${UNITREE_LIB})
set_target_properties(motor_cli PROPERTIES
    INSTALL_RPATH "${SDK_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)

install(TARGETS basic_control motor_cli DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config DESTINATION share/${PROJECT_NAME})

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
