import os
import sys

if __name__ == "__main__":
    try:
        legate_dir = os.environ["LEGATE_DIR"]
    except KeyError:
        print(
            "Please specify the legate installation dir by setting LEGATE_DIR"
        )
        sys.exit(1)
    script_dir = os.path.dirname(os.path.realpath(__file__))
    legion_dir = script_dir + "/legion"
    sys.path.append(legion_dir)
    from jupyter_notebook.install import driver, parse_args

    args, opts = parse_args()
    args.json = os.path.join(script_dir, "legate_jupyter.json")
    args.legion_prefix = legate_dir + "/bin"
    kernel_file_dir = legion_dir + "/" + "jupyter_notebook"
    driver(args, opts, kernel_file_dir)

    # TODO: copy legate_info.py and json file into legate dir
