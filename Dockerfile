## base stage uses NVIDIA's version of RedHat UBI8 with cuda base
## image and then installs python 3.8 and shared dependencies
####################################################################
FROM nvidia/cuda:11.2.1-base-ubi8 as base

# update OS packages and install python3.8 and a few other tools
RUN dnf update --disableplugin=subscription-manager -y && \
    dnf install --disableplugin=subscription-manager -y \
        python38 libgomp procps-ng

# set python3.8 to be the default python version
# otherwise RHEL8 uses python3.6 by default
RUN update-alternatives --set python3 /usr/bin/python3.8

# add a user and grouip to run as instead of root
# Need this for cluster, not EC2
RUN groupadd -g4006 rad_ml && \
    useradd -u105989 -s/bin/bash -grad_ml -md/rad_ml rad_ml && \
    echo 'umask 0077' >> /rad_ml/.bashrc && \
    echo 'set -o vi' >> /rad_ml/.bashrc

# make radptl home dir and shared mount point for external data volumes
RUN mkdir -p /rad_ml/ /mnt/
WORKDIR /rad_ml/

# make sure we have a good search path and that it points to
# /radptl/.local/bin to find the radptl command-line scripts
ENV PATH=/rad_ml/.local/bin:/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin

## build stage builds and installs necessary wheel files
####################################################################
FROM base as train

# install packages needed only for building
RUN dnf install --disableplugin=subscription-manager -y \
        python38-setuptools python38-wheel python38-pip binutils vim-enhanced && \
    pip3 install --upgrade --no-cache-dir pip

# switch to radptl user to make sure wheels are built in home dir
USER rad_ml:rad_ml

# copy radptl source code for building the wheel
RUN mkdir -p /rad_ml/src/ /rad_ml/src/ /mnt/
COPY --chown=rad_ml:rad_ml requirements.txt /rad_ml/src/
COPY --chown=rad_ml:rad_ml src/*.py /rad_ml/src/

# Install required packages
RUN cd src/ && \
    pip3 install -r requirements.txt
