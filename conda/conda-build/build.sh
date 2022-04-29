cp $PREFIX/lib/stubs/libcuda.so $PREFIX/lib/libcuda.so
ln -s $PREFIX/lib $PREFIX/lib64
$PYTHON install.py --cuda --openmp --with-cuda $PREFIX --with-nccl $PREFIX --arch 70,75,80 --install-dir $PREFIX -v
rm $PREFIX/lib/libcuda.so
rm $PREFIX/lib64
# Legion leaves an egg-info file which will confuse conda trying to pick up the information
# Remove it so the legate-core is the only egg-info file added
rm -rf $SP_DIR/legion*egg-info
