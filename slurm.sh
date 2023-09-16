#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <remote_directory_name> <local_directory_path>"
    exit 1
fi

remote_user="lesuthu"
remote_server="beluga.computecanada.ca"
remote_directory_loc="./projects/def-leese196/lesuthu"
remote_directory_name="$1"
local_directory="$2"

#remote_save_dir="$remote_directory_loc""$remote_directory_name"

#ssh "$remote_user@$remote_server" "cd $remote_directory_loc && mkdir $remote_directory_name" 

#ssh "$remote_user@$remote_server" "cd $remote_directory_loc && mkdir $remote_directory_name"

if [ $? -eq 0 ]; then
    echo "Directory '$remote_directory_name' created in '$remote_directory_loc' on '$remote_server'"

    scp -r "$local_directory" "$remote_user@$remote_server:$remote_save_dir" 

    ssh "$remote_user@$remote_server" "mv $remote_directory_loc/$local_directory $remote_directory_loc/$remote_directory_name"

    if [ $? -eq 0 ]; then
        echo "Copied '$local_directory' to '$remote_directory_loc' on '$remote_server'"
    else
        echo "Failed to copy '$local_directory' to '$remote_directory_loc' on '$remote_server'"
    fi
else
    echo "Failed to create directory in '$remote_directory_loc' on '$remote_server'"
fi

