#!/bin/sh
#
# Local build and install of the WElib package
rm dist/*.whl
rm dist/*.gz
python -m build
pip install --force-reinstall dist/*.whl
