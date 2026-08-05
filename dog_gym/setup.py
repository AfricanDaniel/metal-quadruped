from setuptools import find_packages, setup

package_name = 'dog_gym'

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
    description='MuJoCo/Gymnasium sim environment and RL training pipeline for the dog',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'train = dog_gym.train:main',
            'export_policy = dog_gym.export_policy:main',
            'imitation_pretrain = dog_gym.imitation_pretrain:main',
            'verify_belt_decoupling = dog_gym.verify_belt_decoupling:main',
            'manual_motor_control = dog_gym.manual_motor_control:main',
        ],
    },
)
