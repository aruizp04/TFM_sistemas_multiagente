#!/usr/bin/env python3

import argparse
import subprocess
import sys
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory


SENSORS_PLUGIN_NAME = 'gz::sim::systems::Sensors'
SENSORS_PLUGIN_FILENAME = 'libgz-sim-sensors-system.so'


def remove_embedded_sensors_system(urdf_xml):
    root = ET.fromstring(urdf_xml)

    for gazebo in list(root.findall('gazebo')):
        for plugin in gazebo.findall('plugin'):
            if (
                plugin.get('name') == SENSORS_PLUGIN_NAME
                or plugin.get('filename') == SENSORS_PLUGIN_FILENAME
            ):
                root.remove(gazebo)
                break

    return ET.tostring(root, encoding='unicode')


def generate_description(model, namespace):
    turtlebot4_description = get_package_share_directory('turtlebot4_description')
    xacro_file = (
        f'{turtlebot4_description}/urdf/{model}/turtlebot4.urdf.xacro'
    )
    command = [
        'xacro',
        xacro_file,
        'gazebo:=ignition',
        f'namespace:={namespace}',
    ]

    completed = subprocess.run(
        command,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return remove_embedded_sensors_system(completed.stdout)


def main():
    parser = argparse.ArgumentParser(
        description='Generate TurtleBot4 URDF with sensors but without per-model Sensors system.'
    )
    parser.add_argument('--model', default='standard', choices=['standard', 'lite'])
    parser.add_argument('--namespace', default='turtlebot4')
    args = parser.parse_args()

    try:
        sys.stdout.write(generate_description(args.model, args.namespace))
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr)
        return exc.returncode

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
