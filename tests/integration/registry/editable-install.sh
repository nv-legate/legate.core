legate_root=`python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using Legate at $legate_root"
cmake -S . -B build -D legate_core_ROOT=$legate_root
cmake --build build -j 4
python -m pip install -e .
