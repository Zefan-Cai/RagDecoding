export CUDA_VISIBLE_DEVICES=1



base_dir=/mnt/users/v-caizefan/RAGDecoding
setting=20doc
subset=ans


python -m open_instruct.eval.MuSiQue.run_eval \
    --data_dir ${base_dir}/data_${subset}/data_v1 \
    --evaluation_file musique_llama2chat_${setting}.json \
    --save_dir ${base_dir}/results/results_v6_${subset}/ \
    --model ${base_dir}/models/Llama-2-7b-chat-hf/ \
    --tokenizer ${base_dir}/models/Llama-2-7b-chat-hf/ \
    --sample_method first \
    --max_new_tokens 50 \
    --eval_batch_size 1 \
    --load_in_half


# --max_num_examples_per_task 128 \

