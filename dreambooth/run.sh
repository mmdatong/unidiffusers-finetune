export MODEL_NAME="thu-ml/unidiffuser-v1"
export MODEL_NAME="/root/.cache/huggingface/hub/models--thu-ml--unidiffuser-v1/snapshots/89ee5dfe9ebc7b3dd4a87444857487e2348089df"

export INSTANCE_DIR="dog"
export OUTPUT_DIR="outputs"


accelerate launch main.py \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --model_id_or_path $MODEL_NAME \
  --output_dir=$OUTPUT_DIR \
  --lr_warmup_steps=0 \
  --train_text_encoder \
  --max_train_steps=1500

  #--pretrained_model_name_or_path=$MODEL_NAME  \
  #--instance_data_dir=$INSTANCE_DIR \
  #--output_dir=$OUTPUT_DIR \
  #--instance_prompt="a photo of sks dog" \
  #--resolution=512 \

