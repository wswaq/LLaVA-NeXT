export OMP_NUM_THREADS=8
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth0
export NCCL_DEBUG=INFO

LLM_VERSION="lmsys/vicuna-7b-v1.5"
LLM_VERSION_CLEAN="${LLM_VERSION//\//_}"
VISION_MODEL_VERSION="openai/clip-vit-large-patch14-336"
VISION_MODEL_VERSION_CLEAN="${VISION_MODEL_VERSION//\//_}"

############### Pretrain ################

PROMPT_VERSION="v1"

# BASE_RUN_NAME="llavanext-${VISION_MODEL_VERSION_CLEAN}-${LLM_VERSION_CLEAN}-mlp2x_gelu-pretrain_blip558k_plain"
echo "BASE_RUN_NAME: ${BASE_RUN_NAME}"
TOTAL_BATCH_SIZE=128
#if node cnt is 1 set grad acc steps to 2
if [ $NODE_COUNT -eq 1 ]; then
    GRAD_ACC=4
else
    GRAD_ACC=2
fi
BATCH_SIZE=$((((($TOTAL_BATCH_SIZE / $NODE_COUNT)/ $GPU_PER_NODE_COUNT))/ $GRAD_ACC))

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True ACCELERATE_CPU_AFFINITY=1 torchrun --nproc_per_node="${GPU_PER_NODE_COUNT}" --nnodes="${NODE_COUNT}" --node_rank="${RANK}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}" \
    llava/train/train_mem.py \
    --deepspeed scripts/zero3.json \
    --model_name_or_path ${LLM_VERSION} \
    --version ${PROMPT_VERSION} \
    --data_path=/blob/waq/playground/data/data/llava_v1_5_mix665k.json \
    --image_folder /blob/waq/playground/data/ \
    --pretrain_mm_mlp_adapter="./checkpoints/$BASE_RUN_NAME/projectors/mm_projector.bin" \
    --mm_tunable_parts="mm_mlp_adapter,mm_language_model" \
    --mm_vision_tower_lr=2e-6 \
    --vision_tower ${VISION_MODEL_VERSION} \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio square \
    --image_grid_pinpoints "[(336, 672), (672, 336), (672, 672), (1008, 336), (336, 1008)]" \
    --group_by_modality_length True \
    --mm_patch_merge_type spatial_unpad \
    --bf16 True \
    --run_name ${BASE_RUN_NAME}_finetune \
    --output_dir "./checkpoints/${BASE_RUN_NAME}/finetune" \
    --num_train_epochs 1 \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps $GRAD_ACC \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 1e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 32768 \
    --gradient_checkpointing True \
    --dataloader_num_workers 16 \
    --lazy_preprocess True \
    --report_to wandb \
    --torch_compile True \
    --torch_compile_backend "inductor" \
    --dataloader_drop_last True \
    --seed $SEED \
    --attn_implementation flash_attention_2

# You can delete the sdpa attn_implementation if you want to use flash attn
