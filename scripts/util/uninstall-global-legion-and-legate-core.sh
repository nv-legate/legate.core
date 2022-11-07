#! /usr/bin/env bash

rm -rf $(find "$CONDA_PREFIX" -mindepth 1 -type d -name '*realm*') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type d -name '*legion*') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type d -name '*legate*') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type d -name '*Legion*') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type f -name 'realm*.h') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type f -name 'legion*.h') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type f -name 'pygion.py') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type f -name 'legion_top.py') \
       $(find "$CONDA_PREFIX" -mindepth 1 -type f -name 'legion_cffi.py') \
       $(find "$CONDA_PREFIX/lib" -mindepth 1 -type f -name 'librealm*') \
       $(find "$CONDA_PREFIX/lib" -mindepth 1 -type f -name 'libregent*') \
       $(find "$CONDA_PREFIX/lib" -mindepth 1 -type f -name 'liblegion*') \
       $(find "$CONDA_PREFIX/lib" -mindepth 1 -type f -name 'liblgcore*') \
       $(find "$CONDA_PREFIX/lib" -mindepth 1 -type f -name 'legate.core.egg-link') \
       $(find "$CONDA_PREFIX/bin" -mindepth 1 -type f -name '*legion*') \
       $(find "$CONDA_PREFIX/bin" -mindepth 1 -type f -name 'legate') \
       $(find "$CONDA_PREFIX/bin" -mindepth 1 -type f -name 'bind.sh') \
       $(find "$CONDA_PREFIX/bin" -mindepth 1 -type f -name 'lgpatch') \
       ;
