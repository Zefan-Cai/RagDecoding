export CUDA_VISIBLE_DEVICES=0



base_dir=/home/caizf/projects/RagDecoding
setting=0doc


python -m open_instruct.eval.MuSiQue.run_eval \
    --data_dir ${base_dir}/data/data_v2 \
    --evaluation_file musique_llama2chat_${setting}.json \
    --save_dir ${base_dir}/results/results_v2/ \
    --model /home/caizf/models/Llama-2-7b-chat-hf/ \
    --tokenizer /home/caizf/models/Llama-2-7b-chat-hf/ \
    --sample_method first \
    --eval_batch_size 1 \
    --load_in_half


# --max_num_examples_per_task 128 \

