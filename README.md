# OPER: Offline Prioritized Eexperience Replay

This `td` branch performs the TD3+BC case study to demonstrate the effectiveness of OPER.

### Usage
1. Generate or download the OPER-A priority weights. For OPER-R prioriy weights, it can be automatically loaded in the second step.

2. Train TD3+BC on Gym locomotion tasks.

Train on the original dataset
``` 
python main.py --env $env --seed $i --bc_eval=0
```

To reproduce the main results of OPER-A in the paper, i.e., only prioritizing data for policy constraint and improvement terms, train on 5 th prioritized dataset by resampling or resampling:
```
# resample
python main.py --env $env --seed $i --weight_path $PATH --weight_num 3 --iter 5 --std=$STD ---bc_eval=1 --resample --two_sampler

# reweight
python main.py --env $env --seed $i --weight_path $PATH --weight_num 3 --iter 5 --std=2.0 ---bc_eval=1 --reweight --reweight_eval=0
```

To prioritize data for all terms, run the code:
```
# resample
python main.py --env $env --seed $i --weight_path $PATH --weight_num 3 --iter 5 --std=$STD ---bc_eval=1 --resampler

# reweight
python main.py --env $env --seed $i --weight_path $PATH --weight_num 3 --iter 5 --std=2.0 ---bc_eval=1 --reweight
```

To reproduce the main results of OPER-R in the paper, run the code:
```
# resample
python main.py --env $env --seed $i --bc_eval=0 --resample

# reweight
python main.py --env $env --seed $i --bc_eval=0 --reweight
```