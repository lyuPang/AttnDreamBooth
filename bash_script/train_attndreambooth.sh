# Config

# The pre-trained model used in AttnDreamBooth
base_model="stabilityai/stable-diffusion-2-1-base"
# The directory to save the fine-tuned model
output_dir="./models/"
# The directory of the target concept
instance_dir=""
# The category of the target concept
category=""


# Config of stage 1

# Learning rate
ti_learning_rate="1e-3"
# The model will be saved every `save_step` step
ti_save_step=60
# The number of max training step
ti_train_steps=60
# The weight of the mean of the regularization term
ti_attn_mean=0.1
# The weight of the variance of the regularization term
ti_attn_var=0
# Batch size
ti_bs=8
# Gradient accumulation
ti_ga=1
# Validation every `validation_steps` step
ti_validation_steps=30
# The directory to save the output of the first stage
ti_output_dir="${output_dir}/learned_embeds"

# Config of stage 2

attn_learning_rate="2e-5"
attn_save_step=100
attn_train_steps=100
attn_attn_mean=2
attn_attn_var=5
attn_bs=8
attn_ga=1
attn_validation_steps=50

attn_output_dir="${output_dir}/learned_models/attn_models"

# Config of stage 3

unet_learning_rate="2e-6"
unet_save_step=500
unet_train_steps=500
unet_attn_mean=2
unet_attn_var=5
unet_bs=8
unet_ga=1
unet_validation_steps=100

unet_output_dir="${output_dir}/learned_models/unet_models"

# Train
# Stage 1
accelerate launch train_attndreambooth.py \
    --pretrained_model_name_or_path=$base_model \
    --train_ti \
    --save_step=$ti_save_step \
    --max_train_steps=$ti_train_steps \
    --output_dir=$ti_output_dir \
    --object_token="new_concept" \
    --n_object_embedding=1 \
    --initialize_token="${category}" \
    --instance_prompt="a photo of a {} ${category}" \
    --instance_data_dir=$instance_dir \
    --train_batch_size=$ti_bs \
    --gradient_accumulation_steps=$ti_ga \
    --learning_rate=$ti_learning_rate \
    --validation_prompt="a photo of a {} ${category}" \
    --validation_steps=$ti_validation_steps \
    --with_cross_attn_reg \
    --reg_mean_weight $ti_attn_mean \
    --reg_var_weight $ti_attn_var

embedding_path="${ti_output_dir}/learned_embeds_steps_${ti_train_steps}.bin"
# Stage 2
accelerate launch train_attndreambooth.py \
    --pretrained_model_name_or_path=$base_model  \
    --train_cross_attn \
    --only_save_checkpoint \
    --checkpointing_steps=$attn_save_step \
    --max_train_steps=$attn_train_steps \
    --embedding_path="${embedding_path}" \
    --output_dir=$attn_output_dir \
    --resume_from_checkpoint "latest" \
    --instance_data_dir=$instance_dir \
    --instance_prompt="a photo of a {} ${category}" \
    --train_batch_size=$attn_bs \
    --gradient_accumulation_steps=$attn_ga \
    --learning_rate=$attn_learning_rate \
    --validation_prompt="a photo of a {} ${category}" \
    --validation_steps=$attn_validation_steps \
    --with_cross_attn_reg \
    --reg_mean_weight $attn_attn_mean \
    --reg_var_weight $attn_attn_var

# Stage 3
accelerate launch train_attndreambooth.py \
    --pretrained_model_name_or_path=$base_model \
    --train_unet \
    --only_save_checkpoint \
    --checkpointing_steps $unet_save_step \
    --max_train_steps=$unet_train_steps \
    --embedding_path="$embedding_path" \
    --load_from_checkpoint="${attn_output_dir}/checkpoint-${attn_train_steps}" \
    --output_dir=$unet_output_dir \
    --instance_data_dir=$instance_dir \
    --instance_prompt="a photo of a {} ${category}" \
    --train_batch_size=$unet_bs \
    --gradient_accumulation_steps=$unet_ga \
    --learning_rate=$unet_learning_rate \
    --validation_prompt="a photo of a {} ${category}" \
    --validation_steps=$unet_validation_steps \
    --with_cross_attn_reg \
    --reg_mean_weight $unet_attn_mean \
    --reg_var_weight $unet_attn_var