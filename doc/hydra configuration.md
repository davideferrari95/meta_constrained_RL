# Hydra Setup

- Setup Hydra Automatic Activation with `Anaconda`/`Miniconda`
- Automatically `eval` Hydra Scripts when Activate the `conda` Environment

#### Create the Automatic Hydra Activator

- Create a `bash` script `hydra_activate.sh`
- Copy the script in the `etc/conda/activate.d` folder of the desired conda environment.

    `touch ~/miniconda3/envs/**my_env**/etc/conda/activate.d/hydra_activate.sh`

#### Add new Scripts to the Automatic Hydra Activator

- For each script add the corresponding `eval` command:

    `eval "$(python ~/path/to/hydra_script.py -sc install=bash)"`
