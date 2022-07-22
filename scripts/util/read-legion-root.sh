#! /usr/bin/env bash

# Read Legion_ROOT from the environment or prompt the user to enter it
if [[ -z "$Legion_ROOT" ]]; then
    while [[ -z "$Legion_ROOT" || ! -d "$Legion_ROOT" ]]; do
        read -ep "\`\$Legion_ROOT\` not found.
Please enter the path to a Legion build (or install) directory:
" Legion_ROOT </dev/tty
    done
    echo "To skip this prompt next time, run:"
    echo "Legion_ROOT=\"$Legion_ROOT\" $1"
else
    echo "Using Legion at: \`$Legion_ROOT\`"
fi

export Legion_ROOT="$(realpath -m "$Legion_ROOT")"
