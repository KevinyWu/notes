# Foundation Model - Epic Kitchen

## LMN usage

### Directories in ```foundation``` Docker container

- ```$LMN_CODE_DIR: /lmn/foundation/code/```
- ```$LMN_MOUNT_DIR: /lmn/foundation/mount/```
- ```$LMN_OUTPUT_DIR: /lmn/foundation/output/```
- ```$LMN_SCRIPT_DIR: /lmn/foundation/script/```

### Transfer output files in Docker container to local machine

- ```echo 'some-output' > $LMN_OUTPUT_DIR/output_file.txt```
- In local directory, output will be in ```/.output/output/```
  - lmn will copy the output to the local machine when the container is closed
- In remote directory, output will be in ```/scratch/kevinywu/lmn/kevinywu/foundation/output/```
  - This will show up immediately, since the container is running on the remote machine
