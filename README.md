# SDCNLab Solution 1
To run the solution, first clone this repository: \
```git clone https://github.com/York-SDCNLab/Solution1.git```

Then navigate to /path/to/Solution1 and run ```pip install -e .``` to setup \
the local dependencies. Once done, you can install the project dependencies by runnig \
```pip install -r requirements.txt```

Now the project is set up! You can run ```python main.py``` to begin running our \
solution. Note that we have separate processes in ```main.py``` for running the \
traffic light control and map generation. These are identical to the provided \
```Setup_Competition.py``` and ```Traffic_Lights_competition.py``` scripts, however \
if you want to run these scripts manually, you can comment the following lines in main.py.

```
spawn_on_node(node_id=node_id) # spawn the car on the specific node
```

```
traffic_light_process = Process(
    target=run_traffic_light, 
    args=('auto', 1.5, None))
activate_event.clear()
traffic_light_process.start()
time.sleep(4) # wait for the traffic light process to start
```