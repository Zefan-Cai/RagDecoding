export CUDA_VISIBLE_DEVICES=3



base_dir=/home/caizf/projects/RagDecoding
setting=20doc


python -m open_instruct.eval.MuSiQue.run_eval_RAG \
    --data_dir ${base_dir}/data/data_v2 \
    --file_withDoc musique_llama2_${setting}.json \
    --file_withoutDoc musique_llama2_0doc.json \
    --save_dir ${base_dir}/results/results_v2/ \
    --model /home/models/Llama-2-7b-hf/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --max_new_tokens 10 \
    --sample_method first \
    --eval_batch_size 1 \
    --load_in_half


# --max_num_examples_per_task 128 \

