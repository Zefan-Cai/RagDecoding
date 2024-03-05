export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
###
 # @Author: JustBluce 972281745@qq.com
 # @Date: 2023-11-25 14:11:51
 # @LastEditors: JustBluce 972281745@qq.com
 # @LastEditTime: 2024-01-11 11:49:19
 # @FilePath: /ZeroEE/ZeroEE/scripts/PKU_eval_ACE_GenData1000_6definitions_ACE_v2.sh
 # @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
###


base_dir=/home/caizf/projects/RagDecoding

# cot
python -m open_instruct.eval.MuSiQue.run_eval \
    --data_dir ${base_dir}/data \
    --evaluation_file musique_chat_support_doc.jsonl \
    --save_dir ${base_dir}/results/musique_chat_support_doc_llama2chat_test_8GPU/ \
    --model /home/caizf/models/Llama-2-7b-chat-hf/ \
    --tokenizer /home/caizf/models/Llama-2-7b-chat-hf/ \
    --sample_method first \
    --eval_batch_size 8 \
    --max_num_examples_per_task 64 \
    --load_in_half
