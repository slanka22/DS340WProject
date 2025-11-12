## 🛠 Quick Start
Install MetaDrive via:

```bash
git clone https://github.com/metadriverse/metadrive.git
cd metadrive
pip install -e .
```
Follow SUMO installation to install the library:

```
sudo apt-get install git
git clone --recursive https://github.com/eclipse-sumo/sumo
cd sumo
export SUMO_HOME="$PWD"
sudo apt-get install $(cat build_config/build_req_deb.txt build_config/tools_req_deb.txt)
cmake -B build .
cmake --build build -j$(nproc)
```

Launch an experiment for training a vision DQN model for the cologne1 intersection:

```
python experiments/vision_dqn.py  nets/RESCO/cologne1/cologne1.sumocfg cologne1_bev -s 3600 -begin_time 25200  -bev -z 100 
```

Feel free to explore other road networks under `nets` folders.
