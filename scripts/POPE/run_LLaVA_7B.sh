#!/bin/bash

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"
scratch_path='/content/scratch'
CHUNKS=${#GPULIST[@]}


for IDX in $(seq 0 $((CHUNKS-1))); do
    CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
        --model_name LLaVA-7B \
        --model_path ./LLaVA/checkpoints/liuhaotian/llava-v1.5-7b \
        --split val \
        --dataset POPE \
        --prompt mq \
        --theme unanswerable \
        --answers_file ${scratch_path}/output/LLaVA-7B/tmp/${CHUNKS}_${IDX}_l_pope.jsonl \
        --num_chunks $CHUNKS \
        --chunk_idx $IDX \
        --temperature 0.0 \
        --num_beams 1 &
done

wait

output_file=${scratch_path}/output/LLaVA-7B/POPE_val_mq.jsonl

# Clear out the output file if it exists.
> "$output_file"

# Loop through the indices and concatenate each file.
for IDX in $(seq 0 $((CHUNKS-1))); do
    cat ${scratch_path}/output/LLaVA-7B/tmp/${CHUNKS}_${IDX}_l_pope.jsonl >> "$output_file"
    rm ${scratch_path}/output/LLaVA-7B/tmp/${CHUNKS}_${IDX}_l_pope.jsonl
done

# for IDX in $(seq 0 $((CHUNKS-1))); do
#     CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python -m run_model \
#         --model_name LLaVA-7B \
#         --model_path ./LLaVA/checkpoints/liuhaotian/llava-v1.5-7b \
#         --split train \
#         --dataset MAD \
#         --prompt oeh \
#         --theme unanswerable \
#         --answers_file ${scratch_path}/output/LLaVA-7B/tmp/${CHUNKS}_${IDX}_l_mad.jsonl \
#         --num_chunks $CHUNKS \
#         --chunk_idx $IDX \
#         --temperature 0.0 \
#         --top_p 0.9 \
#         --num_beams 1 &
# done

# wait

# output_file=${scratch_path}/output/LLaVA-7B/MAD_train_oeh.jsonl

# # Clear out the output file if it exists.
# > "$output_file"

# # Loop through the indices and concatenate each file.
# for IDX in $(seq 0 $((CHUNKS-1))); do
#     cat ${scratch_path}/output/LLaVA-7B/tmp/${CHUNKS}_${IDX}_l_mad.jsonl >> "$output_file"
#     rm ${scratch_path}/output/LLaVA-7B/tmp/${CHUNKS}_${IDX}_l_mad.jsonl
# done
