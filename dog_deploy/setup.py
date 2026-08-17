from setuptools import find_packages, setup

package_name = 'dog_deploy'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/hardware_bringup.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='danielaugustin2027@u.northwestern.edu',
    description='Sim-to-real bridge: runs a trained policy against the real robot',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'policy_node = dog_deploy.policy_node:main',
            'leg_isolation_test = dog_deploy.leg_isolation_test:main',
            'set_home_and_cache = dog_deploy.set_home_and_cache:main',
            'log_static_observation = dog_deploy.log_static_observation:main',
            'motor_sweep_test = dog_deploy.motor_sweep_test:main',
        ],
    },
)
