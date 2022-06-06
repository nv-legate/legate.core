# Do not compile with NDEBUG until Legion handles it without warnings
export CPPFLAGS="$CPPFLAGS -UNDEBUG"

install_args=()

# We rely on an environment variable to determine if we need to build cpu-only bits
if [ -z "$CPU_ONLY" ]; then
  # cuda, relying on the stub library provided by the toolkit
  install_args+=("--cuda" "--with-cuda" "$BUILD_PREFIX")

  # nccl, relying on the conda nccl package
  install_args+=("--with-nccl" "$PREFIX")

  # targetted architecture to compile cubin support for
  install_args+=("--arch" "70,75,80")
fi

#CPU targeting
install_args+=("--march" "haswell")

#openMP support
install_args+=("--openmp")

# Target directory
install_args+=("--install-dir" "$PREFIX")

# Verbose mode
install_args+=("-v")

# Move the stub library into the lib package to make the install think it's pointing at a live installation
if [ -z "$CPU_ONLY" ]; then
  cp $PREFIX/lib/stubs/libcuda.so $PREFIX/lib/libcuda.so
  ln -s $PREFIX/lib $PREFIX/lib64
fi

echo "Install command: $PYTHON install.py ${install_args[@]}"
$PYTHON install.py "${install_args[@]}"

# Remove the stub library and linking
if [ -z "$CPU_ONLY" ]; then
  rm $PREFIX/lib/libcuda.so
  rm $PREFIX/lib64
fi

# Legion leaves an egg-info file which will confuse conda trying to pick up the information
# Remove it so the legate-core is the only egg-info file added
rm -rf $SP_DIR/legion*egg-info
