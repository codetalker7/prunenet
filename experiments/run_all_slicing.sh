for sparsity_level in 0.1 0.2 0.25 0.3
do
for dataset in wikitext2 #alpaca
do
for model in facebook/opt-6.7b facebook/opt-2.7b facebook/opt-1.3b facebook/opt-125m 
do
for sparsity_technique in bernoulli
do
TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0001 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models2/" \
    --sparsity_technique ${sparsity_technique} \
    --st_checkpoint_dir '../models2/'

TF_CPP_MIN_LOG_LEVEL=2 TF_ENABLE_ONEDNN_OPTS=1 CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python trainable_activation_sparsity.py \
    --log DEBUG \
    --use_gpu \
    --model_name ${model}  \
    --num_episodes 20 \
    --learning-rate-action 0.0001 \
    --sparsity_level ${sparsity_level} \
    --ppl-eval-dataset ${dataset}       \
    --finetune-dataset ${dataset}         \
    --finetune-train-nsamples 8000       \
    --finetune-train-seqlen 1024       \
    --finetune-train-batch-size 3         \
    --lora-alpha 10          \
    --lora-r 32        \
    --lora-dropout 0.05      \
    --lora-target-option attn_head_and_mlp      \
    --eval-steps 16       \
    --save-steps 16 \
    --epochs 1 \
    --model_save_path "../models2/" \
    --finetune \
    --sparsity_technique ${sparsity_technique} \
    --st_checkpoint_dir '../models2/' 
done  
done
done
done
