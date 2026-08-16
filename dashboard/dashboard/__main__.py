"""Lets this package be run directly as `python3 -m dashboard`, matching
this project's own established dev-mode convention (`python3 -m
dog_gym.train`, `python3 -m dog_gym.export_policy`, `python3 -m
dog_deploy.policy_node`) -- in addition to the installed `dashboard`
console-script entry point (`ros2 run dashboard dashboard` after
`colcon build`)."""
from dashboard.app import main

if __name__ == '__main__':
    main()
