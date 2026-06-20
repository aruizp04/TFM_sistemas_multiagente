#!/usr/bin/env bash

# Source this file from any terminal before working with this RMF workspace:
#   source /home/ar_pc/Desktop/TFM/rmf_ws/source_all.bash

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "This script must be sourced, not executed."
  echo "Run: source ${BASH_SOURCE[0]}"
  exit 1
fi

_rmf_ws_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
_ros_distro="${ROS_DISTRO:-jazzy}"
_ros_setup="/opt/ros/${_ros_distro}/setup.bash"
_easynav_ws_dir="${HOME}/easynav_ws"
_easynav_setup="${_easynav_ws_dir}/install/setup.bash"
_ws_setup="${_rmf_ws_dir}/install/setup.bash"
_cyclonedds_config="${_rmf_ws_dir}/cyclonedds_config/cyclonedds.xml"

if [[ ! -f "${_ros_setup}" ]]; then
  echo "ROS setup not found: ${_ros_setup}"
  return 1
fi

if [[ ! -f "${_ws_setup}" ]]; then
  echo "Workspace setup not found: ${_ws_setup}"
  echo "Build the workspace first with: colcon build"
  return 1
fi

source "${_ros_setup}"

if [[ ! -f "${_easynav_setup}" ]]; then
  echo "EasyNav workspace setup not found: ${_easynav_setup}"
  echo "Build it first with: cd ${_easynav_ws_dir} && colcon build"
  return 1
fi

source "${_easynav_setup}"
source "${_ws_setup}"

if ros2 pkg prefix rmw_cyclonedds_cpp >/dev/null 2>&1; then
  export RMW_IMPLEMENTATION="${RMW_IMPLEMENTATION:-rmw_cyclonedds_cpp}"
fi

if [[ -f "${_cyclonedds_config}" ]]; then
  export CYCLONEDDS_URI="${CYCLONEDDS_URI:-file://${_cyclonedds_config}}"
fi

export RMF_WS="${_rmf_ws_dir}"
export EASYNAV_WS="${_easynav_ws_dir}"

echo "Sourced ROS ${ROS_DISTRO:-${_ros_distro}}, EasyNav workspace: ${EASYNAV_WS}"
echo "Sourced RMF workspace: ${RMF_WS}"
echo "RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION:-<default>}"
echo "CYCLONEDDS_URI=${CYCLONEDDS_URI:-<unset>}"

unset _rmf_ws_dir _ros_distro _ros_setup _easynav_ws_dir _easynav_setup _ws_setup _cyclonedds_config
