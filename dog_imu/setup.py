from setuptools import find_packages, setup

package_name = 'dog_imu'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='danielaugustin2027@u.northwestern.edu',
    description='LSM6DSO32 IMU driver node for the Jetson',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'imu_node = dog_imu.imu_node:main',
        ],
    },
)
