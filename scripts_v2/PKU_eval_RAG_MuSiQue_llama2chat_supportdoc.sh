export CUDA_VISIBLE_DEVICES=2



base_dir=/home/caizf/projects/RagDecoding
setting=supportdoc


python -m open_instruct.eval.MuSiQue.run_eval_RAG \
    --data_dir ${base_dir}/data/data_v2 \
    --file_withDoc musique_llama2chat_${setting}.json \
    --file_withoutDoc musique_llama2chat_0doc.json \
    --save_dir ${base_dir}/results/results_v2/ \
    --model /home/caizf/models/Llama-2-7b-chat-hf/ \
    --tokenizer /home/caizf/models/Llama-2-7b-chat-hf/ \
    --max_new_tokens 10 \
    --sample_method first \
    --eval_batch_size 1 \
    --load_in_half


# --max_num_examples_per_task 128 \

