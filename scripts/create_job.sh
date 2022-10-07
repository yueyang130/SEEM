SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

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
USER_ID=$(id -u)
GROUP_ID=$(id -g)
# WANDB_API_KEY=<your key>

CPU="${CPU:-4}"
GPU="${GPU:-1}"
MEM="${MEM:-21Gi}"

PRIORITY="${PRIORITY:-low}"

# If no command given, sleep
if [$ESCAPED_COMMAND = '']; then
  ESCAPED_COMMAND="sleep 86400"
fi
echo $ESCAPED_COMMAND

rm "$TEMPORARY_JOB_FILE"
cp "$TEMPLATE_FILE" "$TEMPORARY_JOB_FILE"
sed -i "s/%{COMMAND}/$ESCAPED_COMMAND/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{USER_ID}/$USER_ID/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{GROUP_ID}/$GROUP_ID/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{USER}/$USER/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{CPU}/$CPU/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{GPU}/$GPU/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{MEM}/$MEM/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{NAMESPACE}/$NS/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{PRIORITY}/$PRIORITY/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{WB_KEY}/$WANDB_API_KEY/g" "$TEMPORARY_JOB_FILE"
sed -i "s/%{SOTA_KEY}/$SOTA_API_KEY/g" "$TEMPORARY_JOB_FILE"

# cat "$TEMPORARY_JOB_FILE"
echo "$NS: $PRIORITY"

kubectl create --namespace $NS -f "$TEMPORARY_JOB_FILE"
