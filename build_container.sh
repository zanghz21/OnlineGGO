sudo singularity build --sandbox singularity/ubuntu_onlineGGO/ singularity/ubuntu_onlineGGO.def
sudo singularity run --writable singularity/ubuntu_onlineGGO
sudo singularity build singularity/ubuntu_onlineGGO.sif singularity/ubuntu_onlineGGO