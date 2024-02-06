model_path=$1
model_name=$2
model_max_len=$3
timestamp=$(date "+%y%m%d%H%M%S")
output_dir="outputs/$model_name/$model_max_len/$timestamp"

echo "output dir $output_dir"

cmd="python3 prediction.py --model-path $model_path --model-name $model_name --model-max-len $model_max_len --output-dir $output_dir --single-process"
echo $cmd
eval $cmd

cmd="python3 evaluation.py --input-dir $output_dir"
echo $cmd
eval $cmd
