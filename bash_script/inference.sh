# This is the path to the embeddings learned in the first stage.
learned_embedding_path=""
# This is the path to the fine-tuned models trained in the third stage.
checkpoint_path=""
# The category of the target concept.
category=""
# The directory where the generated images will be saved.
output_dir=""

# inference
python inference.py \
    --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
    --learned_embedding_path=$learned_embedding_path \
    --checkpoint_path=$checkpoint_path \
    --prompt="A photo of a {}" \
    --category="${category}" \
    --save_dir=$output_dir