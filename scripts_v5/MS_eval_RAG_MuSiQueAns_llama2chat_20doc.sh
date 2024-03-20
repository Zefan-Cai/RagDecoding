export CUDA_VISIBLE_DEVICES=5


base_dir=/mnt/users/v-caizefan/RAGDecoding
setting=20doc
subset=ans



python -m open_instruct.eval.MuSiQue.run_eval_RAG \
    --data_dir ${base_dir}/data_${subset}/data_v1 \
    --file_withDoc musique_llama2chat_${setting}.json \
    --file_withoutDoc musique_llama2chat_${setting}.json \
    --save_dir ${base_dir}/results/results_v6_${subset}/ \
    --model ${base_dir}/models/Llama-2-7b-chat-hf/ \
    --tokenizer ${base_dir}/models/Llama-2-7b-chat-hf/ \
    --max_new_tokens 50 \
    --sample_method first \
    --eval_batch_size 1 \
    --load_in_half

# --max_num_examples_per_task 10 \
# --max_num_examples_per_task 128 \

