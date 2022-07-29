import os
import sys

legion_dir = os.getcwd() + "/legion"
sys.path.append(legion_dir)
from jupyter_notebook.install import driver, parse_args

if __name__ == "__main__":
    try:
        legate_dir = os.environ["LEGATE_DIR"]
    except KeyError:
        print("Please sepecify the legate installtion dir by set LEGATE_DIR")
        sys.exit()
    args, opts = parse_args()
    args.json = "legate_jupyter.json"
    args.legion_prefix = legate_dir
    kernel_file_dir = legion_dir + "/" + "jupyter_notebook"
    driver(args, opts, kernel_file_dir)
