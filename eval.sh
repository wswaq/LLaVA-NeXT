#!/bin/bash
#check NODE_RANK env if equals 0 then run the following code
if [ $NODE_RANK -ne 0 ]; then
    echo "NODE_RANK is not 0, skip"
    exit 0
fi
# 任务列表
tasks=("mme" "ai2d" "chartqa" "docvqa_val" "mathvista_testmini" "mmmu_val" "scienceqa_img" "llava_in_the_wild")
# tasks= ("gqa" "mmvet" "pope" "seed")


# tasks=("ai2d")

ckpt_dir=$BASE_RUN_NAME
echo "ckpt_dir: $ckpt_dir"

# 遍历任务并执行
for task in "${tasks[@]}"; do
  # 定义输出路径
  output_path="/home/aiscuser/waq/instructCLIP/LLaVA-NeXT/checkpoints/$ckpt_dir/logs/${task}"

  # 执行命令
  python3 -m accelerate.commands.launch \
    --num_processes=8 \
    -m lmms_eval \
    --model llava \
    --model_args pretrained="/home/aiscuser/waq/instructCLIP/LLaVA-NeXT/checkpoints/$ckpt_dir/finetune" \
    --tasks $task \
    --batch_size 1 \
    --log_samples \
    --log_samples_suffix $task \
    --output_path $output_path
done
