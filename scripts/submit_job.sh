sed s/%USER/$(whoami)/g ./k8s/job.yaml.template | kubectl --namespace=offrl --cluster=kubernetes create -f -
