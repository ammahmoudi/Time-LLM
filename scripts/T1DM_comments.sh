# Model Configuration
model_name=TimeLLM                # Model name (e.g., TimeLLM, Autoformer, etc.)
llama_layers=1                   # Number of layers in the LLM model
d_model=32                        # Model hidden layer size
d_ff=32                           # Feed-forward network dimension
comment='TimeLLM-T1DM'            # Experiment comment for saving results and logs

# Optimization and Training Settings
train_epochs=1                    # Number of training epochs
learning_rate=0.001               # Learning rate
batch_size=1                      # Batch size for training
eval_batch_size=8                 # Batch size for evaluation/testing
master_port=00098                 # Communication port for distributed training
num_process=1                     # Number of processes for distributed training

# Launch Command with Accelerate
accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  # Task and Training/Testing Configuration
  --task_name long_term_forecast \                  # Task type: defines the type of task (e.g., long-term forecast, classification)
  --is_training 1 \                                 # Whether to run training (1: Train, 0: Skip training)
  --is_testing 1 \                                  # Whether to run testing after training (1: Test, 0: Skip testing)

  # Dataset Configuration
  --root_path ./dataset/T1DM/ \                     # Root directory containing the dataset files
  --train_data_path 570-ws-training.csv \                     # Path to the training dataset file (if separate from testing data)
  --test_data_path 570-ws-testing.csv \                       # Path to the testing dataset file
  --data T1DM \                                     # Dataset name (must match dataset class in data provider)
  --features S \                                    # Forecasting type: 'S' (univariate), 'M' (multivariate), 'MS' (multivariate-to-univariate)
  --target _value \                                 # Target column in the dataset
  --freq 5min \                                     # Data frequency (e.g., 5min, hourly, daily)

  # Input and Output Sequence Lengths
  --seq_len 12 \                                    # Length of input sequence for forecasting
  --label_len 6 \                                   # Start token length for decoder (provides context for prediction)
  --pred_len 12 \                                   # Length of the prediction sequence

  # Model Parameters
  --model_id T1_DM_12_12 \                          # Unique identifier for the experiment (used for logging and checkpoints)
  --model $model_name \                             # Model architecture to use (e.g., TimeLLM, Autoformer, DLinear)
  --d_model $d_model \                              # Size of the model's hidden layers
  --d_ff $d_ff \                                    # Dimension of the feed-forward network in the model
  --factor 1 \                                      # Attention factor for models like Autoformer
  --enc_in 1 \                                      # Input size for the encoder (number of input features)
  --dec_in 1 \                                      # Input size for the decoder
  --c_out 1 \                                       # Output size (number of target features)
  --e_layers 2 \                                    # Number of encoder layers
  --d_layers 1 \                                    # Number of decoder layers
  --n_heads 8 \                                     # Number of attention heads in the model
  --dropout 0.1 \                                   # Dropout rate for regularization
  --moving_avg 25 \                                 # Moving average window size for smoothing
  --activation gelu \                               # Activation function to use (e.g., gelu, relu)
  --embed timeF \                                   # Type of embedding for time features (e.g., timeF, fixed, learned)
  --stride 8 \                                      # Stride size for patch embeddings
  --patch_len 16 \                                  # Length of patches used in embeddings

  # LLM-Specific Parameters
  --llm_layers $llama_layers \                      # Number of layers in the large language model (LLM)
  --llm_model LLAMA \                                # Type of LLM model (e.g., GPT2, LLAMA, BERT)
  --llm_dim 4096 \                                   # Dimension of the LLM model (e.g., hidden size in GPT2)

  # Optimization Parameters
  --batch_size $batch_size \                        # Batch size for training
  --eval_batch_size $eval_batch_size \              # Batch size for evaluation/testing
  --learning_rate $learning_rate \                  # Learning rate for optimization
  --train_epochs $train_epochs \                    # Number of epochs for training
  --patience 10 \                                   # Early stopping patience (stop if validation loss doesn’t improve)
  --use_amp \                                       # Enable automatic mixed precision for faster training (requires `bf16` or `fp16`)
  --lradj COS \                                     # Learning rate adjustment type (e.g., type1, COS for cosine annealing)

  # Experiment Details
  --model_comment $comment \                        # Additional comment or identifier for the experiment
  --prompt_domain 0                                 # Enable domain-specific prompts for models like TimeLLM (0: No, 1: Yes)
