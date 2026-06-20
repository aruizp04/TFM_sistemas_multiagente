# Comandos TurtleBot4 RMF

Ejecutar desde `/home/ar_pc/Desktop/TFM/rmf_ws`.

## Entorno

```bash
source /opt/ros/jazzy/setup.bash
source /home/ar_pc/easynav_ws/install/setup.bash
source install/setup.bash
```

## EasyNav

Terminal 1:

```bash
ros2 launch my_world my_world_tb4.launch.xml navigation_backend:=easynav
```

Terminal 2:

```bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config.yaml \
  -n src/my_world/maps/my_world/nav_graphs/1.yaml \
  -sim \
  --navigation_backend easynav
```

Terminal 3:

```bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config_turtlebot2.yaml \
  -n src/my_world/maps/my_world/nav_graphs/0.yaml \
  -sim
```

Verificacion rapida:

```bash
ros2 topic echo /turtlebot1/localizer_node/costmap/pose --once
ros2 topic echo /turtlebot2/localizer_node/costmap/pose --once
ros2 topic echo /fleet_states --once
```

Patrullas:

```bash
ros2 run rmf_demos_tasks dispatch_patrol \
  -p hall_4 hall_5 \
  -F turtlebot1 \
  -R turtlebot1 \
  -n 1 \
  --use_sim_time
```

```bash
ros2 run rmf_demos_tasks dispatch_patrol \
  -p room5_2 hall_1 \
  -F turtlebot2 \
  -R turtlebot2 \
  -n 1 \
  --use_sim_time
```

## Nav2

Terminal 1:

```bash
ros2 launch my_world my_world_tb4.launch.xml navigation_backend:=nav2
```

Terminal 2:

```bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config.yaml \
  -n src/my_world/maps/my_world/nav_graphs/1.yaml \
  -sim \
  --navigation_backend nav2
```

Terminal 3:

```bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config_turtlebot2.yaml \
  -n src/my_world/maps/my_world/nav_graphs/0.yaml \
  -sim \
  --navigation_backend nav2
```

Verificacion rapida:

```bash
ros2 topic echo /turtlebot1/amcl_pose --once
ros2 topic echo /turtlebot2/amcl_pose --once
ros2 action info /turtlebot1/navigate_to_pose
ros2 action info /turtlebot2/navigate_to_pose
ros2 topic echo /fleet_states --once
```

Las patrullas son las mismas que en EasyNav.
