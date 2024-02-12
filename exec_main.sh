python main.py \
    --seed=555 \
    --env_id=ALE/Breakout-v5 \
    --exp_name=simple_dqn \
    --buffer_size=10000 \
    --total_timesteps=10000000 \
    --start_eps=1.0 \
    --end_eps=1e-2 \
    --exploration_fraction=0.1
