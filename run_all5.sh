#!/bin/bash



sbatch --reservation=ubuntu2204 --nodes=1  --ntasks-per-node=1 --cpus-per-task=9 --gpus-per-task=1 --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=24:00:00 --mem=20G run10.sh 7

#sbatch --reservation=ubuntu2204 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gpus-per-task=1 --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=96:00:00 --mem=40G run10.sh 8

#sbatch --reservation=ubuntu2204 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gpus-per-task=1 --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=96:00:00 --mem=40G run10.sh 9

#sbatch --reservation=ubuntu2204 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --gpus-per-task=1 --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=40:00:00 --mem=30G run10.sh 10

#sbatch --reservation=ubuntu2204 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=30G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=40:00:00 --mem=24G run10.sh 5

#sbatch --reservation=ubuntu1804 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 4

#sbatch --reservation=ubuntu1804 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 3

#sbatch --reservation=ubuntu1804 --nodes=1 --ntasks-per-node=1 --cpus-per-task=9 --mem=16G --mail-user=dane.malenfant@mila.quebec --mail-type=END --time=20:00:00 --mem=24G run10.sh 2
