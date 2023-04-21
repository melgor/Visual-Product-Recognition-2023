#!/bin/bash
set -euo pipefail
IFS=$'\n\t'



TRAIN_ACC_TYPE='nvidia-tesla-v100'
TRAIN_MACHINE_MODE='SPOT'
ZONE="europe-west4-a"
TRAIN_MACHINE_MODE="provisioning-model=${TRAIN_MACHINE_MODE}"


# Required arguments
TASK_ID=$1
EXPERIMENT_TAG=$2


JOB_ID="${TASK_ID}-${EXPERIMENT_TAG}-$(date +%Y%m%d%H%M%S)"
JOB_ID=$(echo "${JOB_ID}" | tr '[:upper:]' '[:lower:]')

if ! [[ "${JOB_ID}-xxxx" =~ ^([a-z]([-a-z0-9]{0,61}[a-z0-9])?)$ ]]
then
  echo "JOB_ID='$JOB_ID' to long"
  exit 1
fi

CMD_LOCAL_FILE="/tmp/${JOB_ID}.txt"
echo "$@" >> "${CMD_LOCAL_FILE}"

PROJECT_ID="useful-gearbox-375210"
IMAGE_PROJECT_ID="${PROJECT_ID}"
BUCKET="gs://ai-crowd/"
BASE_PATH="${BUCKET}/tasks/${TASK_ID}/${JOB_ID}"

JOB_PACKAGES="${BASE_PATH}/vs_cluster/packages/vs-0.1.tar.gz"
TRAIN_STARTUP_SCRIPT="${BASE_PATH}/vs_cluster/train_local_parallel.sh"
TRAIN_MACHINE_TYPE="n1-standard-8"
TRAIN_CLUSTER_ID="${JOB_ID}-trn"


echo -e "\n\n============ releasing package ============"
sleep 1
tar -cf code.tar visual-product-recognition-2023-starter-kit/config visual-product-recognition-2023-starter-kit/visual_search
gsutil cp code.tar "${JOB_PACKAGES}"
gsutil cp scripts/train_local.sh "${TRAIN_STARTUP_SCRIPT}"



echo -e "\n\n============ creating training template ============"
gcloud beta compute instance-templates create "${TRAIN_CLUSTER_ID}" \
  --machine-type "${TRAIN_MACHINE_TYPE}" \
  --accelerator count="${ACC_COUNT}",type="${TRAIN_ACC_TYPE}" \
  --image "${IMAGE_NAME}" \
  --image-project "${IMAGE_PROJECT_ID}" \
  --boot-disk-auto-delete \\
  --"${TRAIN_MACHINE_MODE}" \
  --metadata \
job_id="${JOB_ID}",\
task_id="${TASK_ID}",\
cluster_id="${TRAIN_CLUSTER_ID}",\
startup-script-url="${TRAIN_STARTUP_SCRIPT}",\
job_packages="${JOB_PACKAGES}",\
zone="$ZONE"\
  --scopes \
  https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/cloud.useraccounts.readonly,https://www.googleapis.com/auth/cloudruntimeconfig

echo -e "\n\n============ creating training instance group ============"
gcloud compute instance-groups managed create "${TRAIN_CLUSTER_ID}" \
  --base-instance-name "${TRAIN_CLUSTER_ID}" \
  --size "${NODE_COUNT}" \
  --template "${TRAIN_CLUSTER_ID}" \
  --zone "${ZONE}"