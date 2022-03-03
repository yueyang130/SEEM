sed s/%USER/$(whoami)/g ./k8s/job.yaml.template | kubectl --namespace=game-ai --cluster=kubernetes create -f -
