export PATH="${PATH}:/usr/local/nvidia/bin:/usr/local/cuda/bin"
python main.py \
    --seed=0 \
    --env_id=ENV_ID \
    --exp_name=dqn \
    --record \
    --buffer_size=10000 \
    --plotting_frequency=100 \
    --save_model \
    --total_timesteps=1e7 \
    --start_eps=1.0 \
    --end_eps=1e-2 \
    --exploration_fraction=0.1 \
    --learning_starting_step=8e4 \
    --training_frequency=4 \
    --batch_size=16 \
    --gamma=0.99 \
    --learning_rate=1e-4 \
    --update_frequency=1000
