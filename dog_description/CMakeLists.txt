cmake_minimum_required(VERSION 3.20)
project(dog_description)

find_package(ament_cmake REQUIRED)

install(
        DIRECTORY
        mjcf
        config
        onshape_folders
        launch
        rviz
        DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
