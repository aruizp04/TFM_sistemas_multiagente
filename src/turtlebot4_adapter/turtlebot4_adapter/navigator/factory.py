# Copyright 2021 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Factory for navigation backends."""

from .easynav import EasyNavNavigator
from .nav2 import Nav2Navigator


def create_navigation_backend(name: str, node):
    """Create a navigation backend by name."""
    backend_name = (name or 'nav2').lower()
    if backend_name == 'nav2':
        return Nav2Navigator(node)
    if backend_name == 'easynav':
        return EasyNavNavigator(node)

    raise ValueError(f'Unsupported navigation_backend [{backend_name}]')
