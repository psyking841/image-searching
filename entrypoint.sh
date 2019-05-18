#!/bin/bash
set -e

# If "-e uid={custom/local user id}" flag is not set for "docker run" command, use 9999 as default
USER_ID=${LOCAL_USER_ID:-9001}

# Notify user about the UID selected
echo "Starting with UID : $USER_ID"
# Create user called "docker" with selected UID
useradd --shell /bin/bash -u $USER_ID -o -c "" -m docker
# Set "HOME" ENV variable for user's home directory
export HOME=/home/docker

# nohup python /code/training.py -i /bag_train/*.jpg -o /model &

# Execute process
exec gosu docker "$@"