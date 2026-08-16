from setuptools import find_packages, setup

package_name = 'dashboard'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test', 'dashboard.templates', 'dashboard.static']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    include_package_data=True,
    install_requires=['setuptools', 'flask'],
    zip_safe=True,
    maintainer='daniel',
    maintainer_email='danielaugustin2027@u.northwestern.edu',
    description='Local web control panel for browsing/testing/exporting/deploying dog_gym policies',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'dashboard = dashboard.app:main',
        ],
    },
)
