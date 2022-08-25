#! /usr/bin/env bash

rm -rf $(find "$CONDA_PREFIX" -type d -name '*realm*') \
       $(find "$CONDA_PREFIX" -type d -name '*legion*') \
       $(find "$CONDA_PREFIX" -type d -name '*legate*') \
       $(find "$CONDA_PREFIX" -type d -name '*Legion*') \
       $(find "$CONDA_PREFIX" -type f -name 'realm*.h') \
       $(find "$CONDA_PREFIX" -type f -name 'legion*.h') \
       $(find "$CONDA_PREFIX" -type f -name 'pygion.py') \
       $(find "$CONDA_PREFIX" -type f -name 'legion_top.py') \
       $(find "$CONDA_PREFIX" -type f -name 'legion_cffi.py') \
       $(find "$CONDA_PREFIX/lib" -type f -name 'librealm*') \
       $(find "$CONDA_PREFIX/lib" -type f -name 'libregent*') \
       $(find "$CONDA_PREFIX/lib" -type f -name 'liblegion*') \
       $(find "$CONDA_PREFIX/lib" -type f -name 'liblgcore*') \
       $(find "$CONDA_PREFIX/lib" -type f -name 'legate.core.egg-link') \
       $(find "$CONDA_PREFIX/bin" -type f -name '*legion*') \
       $(find "$CONDA_PREFIX/bin" -type f -name 'legate') \
       $(find "$CONDA_PREFIX/bin" -type f -name 'bind.sh') \
       $(find "$CONDA_PREFIX/bin" -type f -name 'lgpatch') \
       ;
