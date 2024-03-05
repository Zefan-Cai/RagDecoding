export CUDA_VISIBLE_DEVICES=6



base_dir=/home/caizf/projects/RagDecoding
setting=supportdoc


python -m open_instruct.eval.MuSiQue.run_eval \
    --data_dir ${base_dir}/data/data_v2 \
    --evaluation_file musique_llama2_${setting}.json \
    --save_dir ${base_dir}/results/results_v2/ \
    --model /home/models/Llama-2-7b-hf/ \
    --tokenizer /home/models/Llama-2-7b-hf/ \
    --sample_method first \
    --eval_batch_size 1 \
    --load_in_half


# --max_num_examples_per_task 128 \

