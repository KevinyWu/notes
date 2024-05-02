# RIPL Notes

Notes and papers for RIPL projects.

- [RIPL Notes](#ripl-notes)
  - [Project Tree](#project-tree)
  - [Servers](#servers)
  - [Generate SSH Key](#generate-ssh-key)
  - [LMN Usage](#lmn-usage)

## Project Tree

```plaintext
RIPL
├── appendix - classic and general papers
├── foundation - generalist robotic foundation models
├── robot-dynamics-control - notes from "Robot Dynamics and Control, 2nd Edition"
├── robotic-manipulation - notes from "Robotic Manipulation" course at MIT
├── teleoperation - robotic arm teleoperation with Meta Quest
└── video2reward - learning from demonstration with video input
```

## Servers

- Connect to UChicago network or VPN
- RIPL servers
  - ```ssh elm.ttic.edu```
  - ```ssh birch.ttic.edu```
  - Username: kevinywu
- My PC
  - ```ssh maple.ttic.edu```
  - Username: kevinwu
- TTIC cluster: [Beehive](https://beehive.ttic.edu/)
  - ```ssh beehive.ttic.edu```
  - Username: kevinwu
- To close server: ```logout```

## Generate SSH Key

1. Generate key: ```ssh-keygen -t ed25519 -C "kevin.yuanbo@gmail.com"``` (only need to do this once per key)
   - Saves key at ```~/.ssh/<key_name>```
   - In this example, key name is ```ttic_key```
   - Enter empty passphrase
   - Make sure the key permissions look like this with ```ls -la``` (if not, modify them to look like this):
      - ```-rw-------   1 kevinwu  staff   419 Mar 21 19:32 ttic_key```
      - ```-rw-r--r--   1 kevinwu  staff   104 Mar 21 19:32 ttic_key.pub```
2. Copy public key to server: ```scp ~/.ssh/ttic_key.pub kevinywu@elm.ttic.edu:.ssh/```
   - Make sure ```~/.ssh/``` directory exists in the remote server
   - In remote server, copy public key to ```~/.ssh/authorized_keys```: ```cat ~/.ssh/ttic_key.pub >> ~/.ssh/authorized_keys```
     - Note: on the remote server side, files in ```.ssh``` don’t matter. Only those keys that are listed in ```authorized_keys``` file will be used for authentication.
   - Make sure the key permissions look like this with ```ls -la``` (if not, modify them to look like this):
      - ```-rw-rw-r-- 1 kevinywu kevinywu  104 Mar 22 03:17 authorized_keys```
      - ```-rw-r--r-- 1 kevinywu kevinywu  104 Mar 22 03:13 ttic_key.pub```
3. Add key to ssh-agent: ```eval "$(ssh-agent -s)"```
   - In local ```~/.ssh/``` directory: ```ssh-add ttic_key``` (only need to do this once per key)
4. Add host information to config file in your local machine: ```/Users/kevinwu/.ssh/config```

   ```config
   Host elm
      HostName elm.ttic.edu
      IdentityFile ~/.ssh/ttic_key
      User kevinywu
   ```

## LMN Usage

**Launch Docker container in remote machine from local machine: [lmn](https://github.com/takuma-yoneda/lmn)**

- Make sure [rsync is updated](https://dev.to/al5ina5/updating-rsync-on-macos-so-you-re-not-stuck-with-14-year-old-software-1b5i)
- Build the Docker image: (example) ```docker build -t ripl/foundation```
- Add SSH key to remote server (see below)
- Global config: ```/Users/kevinwu/.config/lmn.json5```
- Local config: ```project/.lmn.json5```
- ```lmn run elm -- bash```
- In birch and elm, files located in ```/scratch/kevinywu/lmn/kevinywu``` (can change this location with the global config)
- To exit container: ```ctrl + d```

**Create file in Docker container**

- ```touch foo.txt```
- After exiting container, file will be copied to local and remote machine

**Deleting files in Docker container**

`rsync` will not synchronize deletions, so you must manually delete files in both the container and the local machine.
