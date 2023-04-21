#!/bin/bash
set -euo pipefail
IFS=$'\n\t'

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo "TRAINING: Startup script location ${DIR}"

export PATH=/opt/conda/bin:$PATH
export NCCL_ASYNC_ERROR_HANDLING=1
export GCP_LOGGING_ENABLED="TRUE"

JOB_ID=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/job_id -H "Metadata-Flavor: Google")
TASK_ID=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/task_id -H "Metadata-Flavor: Google")

JOB_PACKAGES=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/job_packages -H "Metadata-Flavor: Google")
ZONE=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/zone -H "Metadata-Flavor: Google")
BASE_PATH=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/base_path -H "Metadata-Flavor: Google")
BASE_CONFIG=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/base_config -H "Metadata-Flavor: Google")
DISKS=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/disks -H "Metadata-Flavor: Google")
VAL_CLUSTER_ID=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/val_cluster_id -H "Metadata-Flavor: Google")

CUSTOM_KWARGS=$(curl --silent http://metadata.google.internal/computeMetadata/v1/instance/attributes/custom_kwargs -H "Metadata-Flavor: Google")
CUSTOM_KWARGS=$(echo "${CUSTOM_KWARGS}" | sed -e 's/^"//' -e 's/"$//')

IFS=' ' read -r -a CUSTOM_KWARGS <<<"${CUSTOM_KWARGS}"
IFS=$'\n\t'


echo -e "TRAINING: job_id: ${JOB_ID},\n packages: ${JOB_PACKAGES}, nodes: ${NODE_COUNT}\n"

PACKAGE_DIR="/tmp/packages"
gsutil cp "${JOB_PACKAGES}" "${PACKAGE_DIR}/"
cd



echo '=========== TRAINING: start  ============'

PYTHONHASHSEED=42 BASE_PATH=$BASE_PATH python -u -m  "${BASE_CONFIG}" \
  --job_id "${JOB_ID}" \
  --node_count "${NODE_COUNT}" \
  "${CUSTOM_KWARGS[@]}"  ||
  (echo '=========== TRAINING: job failed ============' &&
  gcloud compute instance-groups managed delete "${VAL_CLUSTER_ID}" --zone "${ZONE}")


echo -e "\n\n================= TRAINING: cleanning stage ================"
sleep 5
INSTANCE_NAME=$(hostname)
CLUSTER_SIZE=$(gcloud compute instance-groups managed describe "${CLUSTER_ID}" --zone "${ZONE}" --format="text(targetSize)" | cut -d' ' -f2)
echo "TRAINING: ${INSTANCE_NAME}: current cluster ${CLUSTER_ID} size is ${CLUSTER_SIZE}"


echo "TRAINING: deleting instance ${INSTANCE_NAME} from cluster ${CLUSTER_ID}"
gcloud compute instance-groups managed delete-instances "${CLUSTER_ID}" --instances="${INSTANCE_NAME}" --zone "${ZONE}"
sleep 5

echo "TRAINING: to be really sure deleting instance ${INSTANCE_NAME} from cluster ${CLUSTER_ID} again"
gcloud compute instance-groups managed delete-instances "${CLUSTER_ID}" --instances="${INSTANCE_NAME}" --zone "${ZONE}"
