for seed in {0..4}
do
    echo RPG $seed
    python ur5_main.py --config_file configs/mujoco_rpg.json --seed $seed
    killall -u $USER python
done

for seed in {0..4}
do
    echo PPO $seed
    python ur5_main.py --config_file configs/mujoco_ppo.json --seed $seed
    killall -u $USER python
done
