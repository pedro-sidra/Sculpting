#!/bin/sh
# vim:sw=4:ts=4:et
set -e

#apt-get install -y fuse nodejs npm
#npm install -g n
#npm install -g npm@latest
#n 14.0 && hash -r
git config --global --add safe.directory /workspace
python -m pip install -e .
python -m pip install -e ./perlyn
python -m pip install git+https://github.com/isisim/pypcd
cd /workspaces/Sculpting/Pointcept/libs/pointrope && python setup.py install && cd /workspaces/Sculpting

sleep infinity
