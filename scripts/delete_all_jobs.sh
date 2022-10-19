#!/bin/bash

username=$(whoami)
arrIN=(${username//_/ })
username=${arrIN[0]}

kubectl delete job `kubectl get job | grep misa | grep ${username} | grep ${1} | awk '{print $1}'`
