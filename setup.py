from setuptools import setup

setup(
    name='gym_multi_car_racing',
    version='0.0.2',
    url='https://github.com/igilitschenski/multi_car_racing',
    description='Gym Multi Car Racing Environment',
    packages=['gym_multi_car_racing'],
    install_requires=[
        'box2d',
        'shapely',
        'numpy',
        'gymnasium',
        'pygame',
    ]
)