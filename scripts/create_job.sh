SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
LAMBDA_IMAGE="us-docker.pkg.dev/sail-tpu-02/test-lambda/mujoco:latest"
AUTODL_IMAGE="192.168.0.92:5000/max/mujoco:latest"

uname=$(whoami)
CLUSTER="${CLUSTER:-autodl}"
if [ "$CLUSTER" = "autodl" ];
then
  IMAGE=$AUTODL_IMAGE
  IPS="gitlab-cr-pull-secret"
  PROJECT_PATH=$(echo $PWD | sed "s/home\/aiops\/${uname}/workspace/g")
else
  IMAGE=$LAMBDA_IMAGE
  IPS="regcred"
  PROJECT_PATH=$(echo $PWD | sed "s/mnt\/home\/${uname}/workspace/g")
fi

USER=$(whoami)
RELEASE="${RELEASE:-false}"
if [ "$RELEASE" = "true" ];
then
  USER_ID="release"
else
  USER_ID=$USER
fi

TEMPLATE_FILE=$SCRIPT_DIR/../k8s/template.yaml
TEMPORARY_JOB_FILE=/tmp/k8s-job-$USER.yml

echo $TEMPLATE_FILE

# DOCKER_NAME=$1
# DOCKER_TAG=$2
# NameSpace
NS="${NS:-$USER}"

# PROJECT=$DOCKER_NAME
COMMAND="${*:1}"
ESCAPED_COMMAND=$(printf '%s\n' "$COMMAND" | sed -e 's/[\/&]/\\&/g')
GROUP_ID=$(id -g)
WANDB_API_KEY="79a3a5ffa9a9cf664fb64c9eaa2b898ad0915964"

CPU="${CPU:-8}"
GPU="${GPU:-1}"
MEM="${MEM:-32Gi}"

PRIORITY="${PRIORITY:-low}"

# If no command given, sleep
if [$ESCAPED_COMMAND = '']; then
  ESCAPED_COMMAND="sleep 86400"
fi
echo $ESCAPED_COMMAND

ALGO=${ALGO:-MISA}
ALGO_LOWERCASE=${ALGO,,}

rm "$TEMPORARY_JOB_FILE"
cp "$TEMPLATE_FILE" "$TEMPORARY_JOB_FILE"
sed -i "s/%{COMMAND}/$ESCAPED_COMMAND/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{GROUP_ID}/$GROUP_ID/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{USER_ID}/$USER_ID/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{USER}/$USER/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{CPU}/$CPU/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{GPU}/$GPU/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{MEM}/$MEM/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{NAMESPACE}/$NS/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{PRIORITY}/$PRIORITY/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{WB_KEY}/$WANDB_API_KEY/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{SOTA_KEY}/$SOTA_API_KEY/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{ALGO}/$ALGO_LOWERCASE/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{PROJECT_PATH}/${PROJECT_PATH//\//\\/}/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{IMAGE}/${IMAGE//\//\\/}/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{IPS}/${IPS//\//\\/}/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{RELEASE}/${RELEASE//\//\\/}/g" "$TEMPORARY_JOB_FILE"

# cat "$TEMPORARY_JOB_FILE"
echo "$NS: $PRIORITY"
echo $TEMPORARY_JOB_FILE

kubectl create --namespace $NS -f "$TEMPORARY_JOB_FILE"
