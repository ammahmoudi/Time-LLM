model_name=TimeLLM # Model name (e.g., TimeLLM, Autoformer, etc.)
llama_layers=32 # Number of layers in the LLM model
d_model=32 # Model hidden layer size
d_ff=32 # Feed-forward network dimension
comment='TimeLLM-T1DM' #Experiment comment for saving results and logs

# Optimization and Training Settings
train_epochs=2 #Number of training epochs
learning_rate=0.001 # Learning rate
batch_size=7 # Batch size for training
eval_batch_size=8 # Batch size for evaluation/testing
master_port=8388 # Communication port for distributed training
num_process=1 # Number of processes for distributed training

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
--task_name long_term_forecast \
--is_training 1 \
--is_testing 1 \
--root_path ./dataset/T1DM/ \
--train_data_path 570-ws-training.csv \
--test_data_path 570-ws-testing.csv \
--data T1DM \
--features S \
--target _value \
--freq 5min \
--seq_len 12 \
--label_len 6 \
--pred_len 12 \
--model_id T1_DM_12_12 \
--model $model_name \
--d_model $d_model \
--d_ff $d_ff \
--factor 1 \
--enc_in 1 \
--dec_in 1 \
--c_out 1 \
--e_layers 2 \
--d_layers 1 \
--n_heads 8 \
--dropout 0.1 \
--moving_avg 25 \
--activation gelu \
--embed timeF \
--stride 8 \
--patch_len 16 \
--llm_layers $llama_layers \
--llm_model GPT2 \
--llm_dim 768 \
--batch_size $batch_size \
--eval_batch_size $eval_batch_size \
--learning_rate $learning_rate \
--train_epochs $train_epochs \
--patience 10 \
--use_amp \
--lradj COS \
--model_comment $comment \
--prompt_domain 0
