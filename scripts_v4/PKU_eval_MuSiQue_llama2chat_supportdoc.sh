export CUDA_VISIBLE_DEVICES=2



base_dir=/home/caizf/projects/RagDecoding
setting=supportdoc


python -m open_instruct.eval.MuSiQue.run_eval \
    --data_dir ${base_dir}/data/data_v3 \
    --evaluation_file musique_llama2chat_${setting}.json \
    --save_dir ${base_dir}/results/results_v4/ \
    --model /home/caizf/models/Llama-2-7b-chat-hf/ \
    --tokenizer /home/caizf/models/Llama-2-7b-chat-hf/ \
    --sample_method first \
    --max_new_tokens 20 \
    --eval_batch_size 1 \
    --load_in_half


# --max_num_examples_per_task 128 \

