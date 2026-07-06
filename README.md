# Implementación y despliegue de OpenRMF sobre robots móviles

Workspace ROS 2/RMF para simular dos TurtleBot4 en Gazebo y conectarlos con Open-RMF usando dos backends de navegacion:

- `nav2`
- `easynav`

El workspace principal es el `rmf_ws/` que haya creado cada usuario. Ejecuta los comandos desde la raiz de ese workspace salvo que se indique lo contrario.

## Paquetes involucrados

### Paquetes propios

- `my_world`: mundo RMF/Gazebo, mapas, launch principal, configuracion RViz y launch de robots.
  - Launch principal: `src/my_world/launch/my_world_tb4.launch.xml`
  - Launch EasyNav por robot: `src/my_world/launch/easynav_turtlebot4.launch.py`
  - Spawn TurtleBot4: `src/my_world/launch/turtlebot4_spawn.launch.py`
  - Mapa RMF/Gazebo: `src/my_world/maps/my_world/`
  - Grafo RMF `turtlebot1`: `src/my_world/maps/my_world/nav_graphs/1.yaml`
  - Grafo RMF `turtlebot2`: `src/my_world/maps/my_world/nav_graphs/0.yaml`
  - Parametros EasyNav:
    - `src/my_world/config/easynav_turtlebot1.params.yaml`
    - `src/my_world/config/easynav_turtlebot2.params.yaml`

- `turtlebot4_adapter`: fleet adapter RMF para los TurtleBot4.
  - Entry point: `ros2 run turtlebot4_adapter fleet_adapter`
  - Config flota `turtlebot1`: `src/turtlebot4_adapter/config.yaml`
  - Config flota `turtlebot2`: `src/turtlebot4_adapter/config_turtlebot2.yaml`
  - Backends implementados:
    - `src/turtlebot4_adapter/turtlebot4_adapter/navigator/nav2.py`
    - `src/turtlebot4_adapter/turtlebot4_adapter/navigator/easynav.py`

### Paquetes externos usados por el launch

- `rmf_demos`: arranque comun de RMF, schedule, dispatcher y RViz.
- `rmf_demos_gz`: simulacion Gazebo para mundos RMF.
- `rmf_demos_fleet_adapter`: adapter de demostracion `tinyRobot`, lanzado junto al mundo.
- `rmf_demos_tasks`: envio de tareas RMF de prueba.
- `turtlebot4_gz_bringup`: nodos y bridges del TurtleBot4 simulado.
- `turtlebot4_navigation`: localizacion y Nav2 para TurtleBot4.
- `irobot_create_*`: descripcion, control y nodos Create3.
- `ros_gz_sim` y `ros_gz_bridge`: Gazebo y bridges ROS/Gazebo.
- `easynav_system`, `easynav_goalmanager_py` y `easynav_support_py`: backend EasyNav.

## Preparar entorno

Primero compila el workspace:

```bash
cd /ruta/al/rmf_ws
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
```

Despues del build, la forma recomendada de preparar cada terminal es usar el helper del repositorio:

```bash
source source_all.bash
```

Este script carga ROS 2 Jazzy, el overlay de EasyNav en `~/easynav_ws`, el overlay de `rmf_ws`, y configura las variables DDS del workspace cuando estan disponibles. Si el workspace todavia no esta compilado, el script avisara de que falta `install/setup.bash`.

Para usar `navigation_backend:=easynav`, EasyNav debe estar instalado y compilado en `~/easynav_ws`, porque `source_all.bash` busca exactamente:

```bash
~/easynav_ws/install/setup.bash
```

Si EasyNav esta en otra ruta, mueve o clona ese workspace a `~/easynav_ws` antes de usar los comandos de EasyNav.

Si solo estas iterando sobre el adapter:

```bash
cd /ruta/al/rmf_ws
source /opt/ros/jazzy/setup.bash
colcon build --packages-select turtlebot4_adapter --symlink-install
source source_all.bash
```

## Lanzar simulacion con Nav2

Terminal 1:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 launch my_world my_world_tb4.launch.xml navigation_backend:=nav2
```

`nav2` es el valor por defecto, por lo que este comando es equivalente:

```bash
ros2 launch my_world my_world_tb4.launch.xml
```

Este launch arranca:

- RMF core y RViz.
- Gazebo con `my_world`.
- El adapter demo `tinyRobot`.
- `turtlebot1` y `turtlebot2`.
- Localizacion AMCL y Nav2 para ambos TurtleBot4.

## Lanzar flotas con Nav2

Abre una terminal por flota despues de levantar la simulacion.

Terminal 2, flota `turtlebot1`:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config.yaml \
  -n src/my_world/maps/my_world/nav_graphs/1.yaml \
  -sim \
  --navigation_backend nav2
```

Terminal 3, flota `turtlebot2`:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config_turtlebot2.yaml \
  -n src/my_world/maps/my_world/nav_graphs/0.yaml \
  -sim \
  --navigation_backend nav2
```

## Lanzar simulacion con EasyNav

Terminal 1:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 launch my_world my_world_tb4.launch.xml navigation_backend:=easynav
```

Este launch arranca:

- RMF core y RViz.
- Gazebo con `my_world`.
- Bridge explicito de `/clock`.
- El adapter demo `tinyRobot`.
- `turtlebot1` y `turtlebot2`.
- `easynav_system/system_main` por cada TurtleBot4, usando parametros separados.

En modo EasyNav no se debe levantar Nav2 para los TurtleBot4.

## Lanzar flotas con EasyNav

Abre una terminal por flota despues de levantar la simulacion.

Terminal 2, flota `turtlebot1`:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config.yaml \
  -n src/my_world/maps/my_world/nav_graphs/1.yaml \
  -sim \
  --navigation_backend easynav
```

Terminal 3, flota `turtlebot2`:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 run turtlebot4_adapter fleet_adapter \
  -c src/turtlebot4_adapter/config_turtlebot2.yaml \
  -n src/my_world/maps/my_world/nav_graphs/0.yaml \
  -sim \
  --navigation_backend easynav
```

## Enviar tareas RMF de prueba

Ejemplo de bucle para `turtlebot1`:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 run rmf_demos_tasks dispatch_loop \
  -s turtlebot1_charger \
  -f hall_5 \
  -n 1 \
  --use_sim_time
```

Ejemplo de bucle para `turtlebot2`:

```bash
cd /ruta/al/rmf_ws
source source_all.bash
ros2 run rmf_demos_tasks dispatch_loop \
  -s turtlebot2_charger \
  -f room5_2 \
  -n 1 \
  --use_sim_time
```

## Notas de aislamiento entre backends

- Cambios para EasyNav deben mantenerse en bloques o archivos especificos de EasyNav.
- Cambios para Nav2 no deben modificar parametros ni remappings de EasyNav salvo que sea intencionado.
- El selector del launch principal es `navigation_backend`.
- El selector del fleet adapter es `--navigation_backend`.
- Si no se pasa `--navigation_backend`, el adapter usa `fleet_manager.navigation_backend` del YAML o `nav2` como valor por defecto.
