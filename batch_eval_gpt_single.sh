model_name=$1
model_max_len=$2
timestamp=000000000000
output_dir="outputs/$model_name/$model_max_len/$timestamp"

echo "output dir $output_dir"

cmd="python3 prediction_gpt.py --model-name $model_name --model-max-len $model_max_len --output-dir $output_dir"
echo $cmd
eval $cmd

cmd="python3 evaluation.py --input-dir $output_dir"
echo $cmd
eval $cmd
