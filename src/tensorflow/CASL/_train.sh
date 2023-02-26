#!/bin/bash
function print_header(){
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
    echo $1
    printf '%*s\n' "${COLUMNS:-$(tput cols)}" '' | tr ' ' -
}

# Check for libtcmalloc_minimal.so.4                                                                           
if [[ ! -f /usr/lib/libtcmalloc_minimal.so.4 ]] 
then
    echo 'File "/usr/lib/libtcmalloc_minimal.so.4 does not exist, aborting. Make sure you ran dependencies_inst    all.sh if using ALE code!'
    exit
else
    if [[ $LD_PRELOAD != *"/usr/lib/libtcmalloc_minimal.so.4"* ]]; then
        echo '$LD_PRELOAD does not contain "/usr/lib/libtcmalloc_minimal.so.4", aborting. Make sure you ran dep    endencies_install.sh if using ALE code!'
        exit
    fi
fi

# Directory of this script
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )" 

# Doorpuzzle python module                                                                              
print_header "Installing Doorpuzzle python module"
cd $DIR/../../environment/Doorpuzzle
sudo pip install -I .

# Minecraft python module                                                                              
print_header "Installing Minecraft python module"
cd $DIR/../../environment/Minecraft
sudo pip install -I .

# Train tf 
print_header "Training network"
cd $DIR
python CASL.py "$@" 
