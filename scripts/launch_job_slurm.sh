#!/bin/bash

export $(env | grep -v "BASH_" | cut -d= -f1)

PARTITION="${PARTITION:-RTX3090}"
NODE="${NODE:-node06}"

file_num=0
command_num=0
file_prefix=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1) # 生成 6 位随机前缀
file_dir="./slurm_batch"

# 如果 slurm_batch 文件夹不存在，则创建
if [ ! -d "$file_dir" ]; then
    mkdir "$file_dir"
fi

# 运行 launch_job.sh，并将每两条命令放到一个文件中
./scripts/launch_job.sh | while read -r command; do
    echo "$command" >> "${file_dir}/${file_prefix}_${file_num}.sh"
    ((command_num++))
    if [ $((command_num%2)) -eq 0 ]; then
        echo "wait" >> "${file_dir}/${file_prefix}_${file_num}.sh"
        chmod +x "${file_dir}/${file_prefix}_${file_num}.sh"
        ((file_num++))
    fi
done

# 如果最后只有一条命令，需要将该命令加入到最后一个文件中
if [ $((command_num%2)) -ne 0 ]; then
    chmod +x "${file_dir}/${file_prefix}_${file_num}.sh"
fi

# 使用 srun 命令运行所有以 "${file_prefix}" 为前缀的 .sh 文件
for file in "${file_dir}/${file_prefix}"_*.sh; do
  srun -J offbench -N 1 -p $PARTITION -w $NODE --gres gpu:1 bash "$file" &
done

echo "生成的文件前缀为 ${file_prefix}"
wait