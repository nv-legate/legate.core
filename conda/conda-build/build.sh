# Install legate_core C++ libs
tar -C "$PREFIX" --exclude="*.a" --strip-components=1 -xf /tmp/out/legate_core-*-Linux.tar.gz;
sed -E -i "s@/home/coder/\.conda/envs/legate@$PREFIX@g" "$PREFIX/share/Legion/cmake/LegionConfigCommon.cmake";
sed -E -i "s@/home/coder/legate/build/_CPack_Packages/Linux/TGZ/legate_core-(.*)-Linux@$PREFIX@g" "$SP_DIR/legion_canonical_cffi.py";

# Install legate_core Python wheel
pip install --no-deps --root / --prefix "$PREFIX" /tmp/out/legate.core-*.whl;

# Legion leaves .egg-info files, which confuses conda trying to pick up the information
# Remove them so legate-core is the only egg-info file added.
rm -rf "$SP_DIR"/legion*egg-info;
