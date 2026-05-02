ros2 run rmf_building_map_tools building_map_generator gazebo \
  my_world.building.yaml \
  my_world.world \
  models

ros2 run rmf_building_map_tools building_map_generator nav \
  my_world.building.yaml \
  nav_graphs

ros2 run rmf_building_map_tools building_map_generator navgraph_visualization \
  my_world.building.yaml \
  navgraph_visualization
