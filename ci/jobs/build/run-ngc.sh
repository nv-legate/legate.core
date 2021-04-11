#!/bin/bash -l

set -e
set -x
NUMARGS=$#
ARGS=$*

# Logger function for build status output
function logger() {
  echo -e "\n>>>> $@\n"
}

# Arg parsing function
function hasArg {
    (( ${NUMARGS} != 0 )) && (echo " ${ARGS} " | grep -q " $1 ")
}

# Create a workspace for this build
[[ $(ngc workspace create) =~ ID:[[:space:]]+([^[:space:]\']+) ]]
export LEGATE_WORKSPACE=${BASH_REMATCH[1]}
# Extract working tree into a directory
git archive --prefix=repository-archive/legate.core/ HEAD | tar x
# upload the working tree to the workspace
ngc workspace upload --source repository-archive $LEGATE_WORKSPACE
# run the build (using -j 8 for the dgx1v.16g.1.norm instance, otherwise 80 threads detected)
NGC_BATCH_OUTPUT=$(ngc batch run --name "legate-core-image-build" --image "nvcr.io/nvidian/legion/kaniko-project-executor-with-bin:debug" --ace nv-us-west-2 --instance dgx1v.16g.1.norm --commandline "echo '{\"auths\": {\"nvcr.io\": {\"auth\": \"${DOCKER_AUTH_TOKEN}\"}}}'>/kaniko/.docker/config.json && executor --dockerfile legate.core/docker/Dockerfile --destination nvcr.io/nvidian/legion/legate.core:${CI_COMMIT_SHORT_SHA} --destination nvcr.io/nvidian/legion/legate.core:latest --build-arg INSTALL-ARGS=\"-j 8\"" -w ${LEGATE_WORKSPACE}:/workspace/ --result /result --org nvidian --team legion --waitend)
# Get the job ID
[[ $NGC_BATCH_OUTPUT =~ Id:[[:space:]]+([^[:space:]]+) ]]
export NGC_JOB_ID=${BASH_REMATCH[1]}
# Download the result
mkdir ngc-artifacts
cd ngc-artifacts
ngc result download $NGC_JOB_ID
cd ..
# Remove workspace
ngc workspace remove -y $LEGATE_WORKSPACE 

# Remove the result when not needed anymore
# Skip this step for now
# ngc result remove -y $NGC_JOB_ID

# Dump the result into the job log
cat ngc-artifacts/$NGC_JOB_ID/joblog.log
# check if the job was successfully completed
[[ $NGC_BATCH_OUTPUT =~ "Status: FINISHED_SUCCESS" ]]