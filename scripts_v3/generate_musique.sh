
dataset="musique"

dataset_mode="full"
service="pku"
base_dir="/mnt/users/v-caizefan/RAGDecoding"
input_dir="MuSiQue/musique_"${dataset_mode}"_v1.0_dev.jsonl"
output_dir="data_"${dataset_mode}"/data_v4"


strArray=

# 遍历数组中的每个元素
for model in "llama2" "llama2chat"
    do
    for setting in "0doc" "20doc" "supportdoc"
        do
            output_name=${dataset}"_"${model}"_"${setting}".json"
            python ./generate_MuSiQue.py \
                --service ${service} \
                --model ${model} \
                --setting ${setting} \
                --base_dir ${base_dir} \
                --input_dir ${input_dir} \
                --output_dir ${output_dir} \
                --output_name ${output_name}
        done
    done

