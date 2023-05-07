#!/bin/bash

export $(env | grep -v "BASH_" | cut -d= -f1)

PARTITION="${PARTITION:-RTX2080Ti}"
# NODE="${NODE:-node08}"
PROC="${PROC:-2}"

file_num=0
command_num=0
file_prefix=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 6 | head -n 1) # 生成 6 位随机前缀
file_dir="./slurm_batch"

# 如果 slurm_batch 文件夹不存在，则创建
if [ ! -d "$file_dir" ]; then
    mkdir "$file_dir"
fi

# 运行 launch_job.sh，并将每两条命令放到一个文件中
bash launch_job.sh | while read -r command; do
    echo "$command" >> "${file_dir}/${file_prefix}_${file_num}.sh"
    ((command_num++))
    if [ $((command_num%$PROC)) -eq 0 ]; then
        ((file_num++))
    fi
done


# 使用 srun 命令运行所有以 "${file_prefix}" 为前缀的 .sh 文件
for file in "${file_dir}/${file_prefix}"_*.sh; do
    line_count=$(wc -l < $file)
    if [ "$line_count" -eq 1 ] && [ "$PARTITION" == RTX3090 ]; then
        head -n 1 $file >> $file
    fi
    echo "wait" >> "$file"
    chmod +x "$file"    
#   srun -J offbench -N 1 -p $PARTITION -w $NODE --gres gpu:1 bash "$file" &
  srun -J offbench -N 1 -p $PARTITION --exclude=node09 --gres gpu:1 bash "$file" &
done

echo "生成的文件前缀为 ${file_prefix}"