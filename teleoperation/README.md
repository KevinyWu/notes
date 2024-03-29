# UR5 Teleoperation with Meta Quest 3

## Meta Account Login

- Gmail
  - [ripl.ttic@gmail.com](mailto:ripl.ttic@gmail.com)
  - Password: ver*ybv9zav.vzv9DYZ
- Meta
  - [ripl.ttic@gmail.com](mailto:ripl.ttic@gmail.com)
  - Password: evq3mkm5xjx2xkm@AMK (but normally it just asks for an authorization code in Gmail)
  - If ever asked for a 4-digit pin, it will be 6045 (TTIC address)

**Activate developer mode**: [instructions](https://www.linkedin.com/pulse/how-easily-activate-developer-mode-your-quest-3-headset-satya-dev-qxetc/)

## Wifi on Bone Server

Wifi setup on Ubuntu 20.04 with Realtek Semiconductor 802.11ac NIC wifi adapter. The headset and the ROS server need to be on the same wifi network.

Instructions from [this repo](https://github.com/brektrou/rtl8821CU) and [this post](https://askubuntu.com/questions/1162974/wireless-usb-adapter-0bdac811-realtek-semiconductor-corp).

1. Check the wifi adapter is connected: `lsusb`, and look for `Realtek Semiconductor Corp. 802.11ac NIC`
2. Clone the repo

    ``` bash
    mkdir -p ~/build
    cd ~/build
    git clone https://github.com/brektrou/rtl8821CU.git
    cd rtl8821CU
    ```

3. Install the driver

    ``` bash
    sudo apt update
    sudo apt install build-essential git dkms bc
    chmod +x dkms-install.sh
    sudo ./dkms-install.sh
    sudo modprobe 8821cu
    ```

4. Check that new network has wireless capability: `sudo lshw -C network`, and see the logical name (should be `wlx001325ae32ed`)
5. IP address: `ifconfig`, then get the `inet` under the logical name

## Quest2ROS

[Project page](https://quest2ros.github.io/)

1. Get the [Quest2ROS app](https://www.oculus.com/experiences/7012450898797913/release-channels/901395321545267/?token=OL4XYlhd) (needs developer mode activated) and download onto headset
2. Clone [ROS TCP enpoint](https://github.com/Unity-Technologies/ROS-TCP-Endpoint) and quest2ros into your catkin workspace src, then build

    ``` bash
    cd ~/workspace/catkin_ws/src
    git clone https://github.com/Unity-Technologies/ROS-TCP-Endpoint.git
    git clone https://github.com/Quest2ROS/quest2ros.git
    catkin build
    ```

3. Make sure ROS server and headset are on the same wifi network
4. From `~/workspace/catkin_ws/src`, start ROS TCP endpoint at the IP address of the ROS PC (found using `ifconfig`, see previous section)

    `roslaunch ros_tcp_endpoint endpoint.launch tcp_ip:=192.168.0.90 tcp_port:=10000`

5. Open Quest2ROS in the headset and set the IP to `192.168.0.90` and the same port
6. Run ros2quest demo

    `rosrun quest2ros ros2quest.py`

7. You can now move the dice and Q2R logo in the VR by pressing the X/A button of left/right controller, respectively
