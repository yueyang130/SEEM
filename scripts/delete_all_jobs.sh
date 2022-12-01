#!/bin/bash

username=$(whoami)
NS="${NS:-$username}"
RELEASE="${RELEASE:-False}"
if [ "$RELEASE" = "true" ];
then
  USER_ID="release"
else
  USER_ID=${username}
fi

kubectl delete job -n ${NS} `kubectl get job -n ${NS} | grep off | grep $USER_ID | awk '{print $1}'`
