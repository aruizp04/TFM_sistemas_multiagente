# Repository Guidelines

## Project Structure & Module Organization

This repository is a ROS 2/RMF workspace. Active source packages live in `src/`.
Generated artifacts in `build/`, `install/`, and `log/` should not be edited
directly. The main custom packages are `src/turtlebot4_adapter/` for the Python
fleet adapter and `src/my_world/` for launch files, configuration, maps, and
Gazebo/RMF world assets. Experiment outputs are stored in `results/`.

## Build, Test, and Development Commands

Run commands from this directory unless noted.

```bash
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
colcon test --packages-select turtlebot4_adapter
colcon test-result --verbose
ros2 launch my_world my_world_tb4.launch.xml
ros2 run turtlebot4_adapter fleet_adapter --config_file src/turtlebot4_adapter/config.yaml
```

Use package-specific builds, for example
`colcon build --packages-select turtlebot4_adapter`, when iterating on one
component.

## Coding Style & Naming Conventions

Python packages use ROS 2 `ament_python` conventions: 4-space indentation,
`snake_case` modules/functions, and package entry points in `setup.py`. Keep
launch files in `launch/`, YAML configuration in `config/` or package roots
where existing setup files install them, and maps/assets under `maps/`. C++
packages follow ROS 2 CMake layout with headers in `include/`, sources in
`src/`, and tests in `test/`. Do not commit generated `__pycache__`, `build/`,
`install/`, or `log/` contents.

## Testing Guidelines

The `turtlebot4_adapter` package includes `pytest`-based style tests:
`test_flake8.py`, `test_pep257.py`, and `test_copyright.py`. Name Python tests
`test_*.py` and place them in the package `test/` directory. After changes, run
targeted `colcon test --packages-select <package>` first, then inspect failures
with `colcon test-result --verbose`.

## Configuration & Safety Notes

Treat `config.yaml`, `cyclonedds_config/cyclonedds.xml`, launch files, and map
assets as environment-sensitive. Document robot names, fleet names, DDS
settings, and map changes so other contributors can reproduce the same
simulation or hardware setup.

Keep each `navigation_backend` isolated from the others. Changes applied for
`easynav` must not alter `nav2` behavior unless explicitly required, and
changes applied for `nav2` must not alter `easynav` behavior. Prefer
backend-gated launch blocks, backend-specific parameter files, and
backend-specific remappings so fixes for one stack do not regress the other.

When a new ROS 2, RMF, Gazebo, TurtleBot4, navigation, TF, DDS, build, or launch
error is identified while working in this repository, update
`../ERRORES_Y_SOLUCIONES.md` automatically as part of the same task. Add a new
entry or extend an existing one with the symptom, where it appeared, the command
that reproduced it, the likely cause, the applied solution or current
workaround, related files, verification commands, and any pending follow-up. Do
not wait for a separate request unless the user explicitly asks not to edit
documentation.

## Restore Checkpoint Stash

On 2026-06-17, the workspace was reset to a clean Git state on branch
`3robots_easynav` at commit `558d87f` (`Cambio de escala, 1 tiny y 2
turtlebot`). Before resetting, the current EasyNav/RMF integration work was
saved in the main workspace stash:

```bash
cd /home/ar_pc/Desktop/TFM/rmf_ws
git stash list
git stash apply stash@{0}
```

The expected stash message is:

```text
On jazzy: checkpoint before documentation reset - easynav integration
```

Use `git stash apply` to recover the changes while keeping the stash available.
Use `git stash pop` only if the user explicitly wants to remove the recovery
point after applying it.

Two nested repositories also received cache-only stashes:

```bash
cd /home/ar_pc/Desktop/TFM/rmf_ws/src/demonstrations/rmf_demos
git stash list

cd /home/ar_pc/Desktop/TFM/rmf_ws/src/rmf/rmf_visualization
git stash list
```

Those nested stashes only contained generated `__pycache__` files and usually do
not need to be restored.

The directories `build/`, `install/`, and `log/` were intentionally kept on disk
so the workspace could be reused without recompiling everything. They are
excluded locally through `.git/info/exclude`, not through a committed
`.gitignore`.
