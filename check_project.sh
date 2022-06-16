#!/bin/#!/usr/bin/env bash
clear

#needs to be implemented, this script checks if dataset exists and let you run the implementation you want
datasets_dir=../datasets_dir
dataset_downloaded=../datasets_dir/ROD-synROD
ROD_dir=../datasets_dir/ROD-synROD/ROD
synROD_dir=../datasets_dir/ROD-synROD/synROD

if [ -d "$datasets_dir" ]
then
  #echo "$datasets_dir is a directory."
  if [ -d "$dataset_downloaded" ]
  then
    #echo "Dataset has been downloaded"
    if [ -d "$ROD_dir" ]
    then
      #echo "Dataset ROD is present"
      if [ -d "$synROD_dir" ]
      then
        #echo "Dataset synROD is present"
        echo "What do you want to run?"
        printf "1 -> original implementation \n2 -> multiTask implementation \n3 -> bottlNeck implementation (not working): "
        read choice

        if (( choice == "1" ));then
          bash ./Implementation/run.sh
          exit 0
        else
          if (( choice == "2" ));then
            bash ./multipleTasks/run.sh
            exit 0
          else
            echo "Implementation of a bottleneck is still not finished"
            exit 0
          fi
        fi








      fi
    fi
  fi
fi

exit 0
