
dataset="musique"

service="pku"
base_dir="/home/caizf/projects/RagDecoding"
input_dir="MuSiQue/musique_full_v1.0_dev.jsonl"
output_dir="data/data_v3"


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

