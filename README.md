# User Simulation
This document describes how to run the simulation and agents (rule, command line, RL).
## Data
under this folders: ./src/deep_dialog/data

* Movie Knowledge Bases<br/>
`movie_kb.1k.p` --- 94% success rate (for user_goals_first_turn_template_subsets.v1.p)<br/>
`movie_kb.v2.p` --- 36% success rate (for user_goals_first_turn_template_subsets.v1.p)

* User Goals<br/>
`user_goals_first_turn_template.v2.p` --- first turn<br/>
`user_goals_first_turn_template.part.movie.v1.p` --- a subset of user goal [Please use this one, the upper bound success rate on movie_kb.1k.json is 0.9765.]

* NLG Rule Template<br/>
`dia_act_nl_pairs.v6.json` --- some predefined NLG rule templates for both User simulator and Agent.

* Dialog Act Intent<br/>
`dia_acts.txt`

* Dialog Act Slot<br/>
`slot_set.txt`

## Parameter

`--agt`: the agent id
`--usr`: the user (simulator) id
`--max_turn`: maximum turns
`--episodes`: how many dialogues you want to run
`--slot_err_prob`: slot level err probability
`--slot_err_mode`: which kind of slot err mode
`--intent_err_prob`: intent level err probability

`--movie_kb_path`: the movie kb path for agent side
`--goal_file_path`: the user goal file path for user simulator side

`--dqn_hidden_size`: hidden size for RL (DQN) agent
`--batch_size`: batch size for DQN training
`--simulation_epoch_size`: how many dialogue to be simulated in one epoch

`--warm_start`: use rule policy to fill the experience replay buffer at the beginning.
`--warm_start_epochs`: how many dialogues to run in the warm start

`--run_mode`: 0 for display mode (NL); 1 for debug mode (dia_act); 2 for debug mode (dia_act and NL); >3 for no display (i.e. training)
`--auto_suggest`: 0 for no auto_suggest; 1 for auto_suggest.
`--act_level`: 0 for user simulator is dia_act level; 1 for user simulator is NL level
`--cmd_input_mode`: 0 for NL input; 1 for Dia_Act input. (this is for AgentCmd only)

`--write_model_dir`: the directory to write the models
`--trained_model_path`: the trained RL agent model; load the trained model for prediction purpose.


## Tutorial for Running Different Agents and User Simulators

### Rule Agent
```sh
python run.py --agt 5 
              --usr 1
	      --max_turn 40
	      --episodes 150
	      --movie_kb_path .\deep_dialog\data\movie_kb.1k.p
	      --goal_file_path .\deep_dialog\data\user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
```

### Cmd Agent
NL Input:
```sh
python run.py --agt 0
              --usr 1
	      --max_turn 40
	      --episodes 150
	      --movie_kb_path .\deep_dialog\data\movie_kb.1k.p
	      --goal_file_path .\deep_dialog\data\user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
	      --run_mode 0
	      --cmd_input_mode 0
```
Dia_Act Input
```sh
python run.py --agt 0
	      --usr 1
	      --max_turn 40
	      --episodes 150
	      --movie_kb_path .\deep_dialog\data\movie_kb.1k.p 
	      --goal_file_path .\deep_dialog\data\user_goals_first_turn_template.part.movie.v1.p
	      --intent_err_prob 0.00
	      --slot_err_prob 0.00
	      --episodes 500
	      --act_level 0
	      --run_mode 0
	      --cmd_input_mode 1
```
Train RL Agent
[End2End without NLU and NLG, with simulated noise in NLU]
```sh
python run.py --agt 9
	      --usr 1
	      --max_turn 40
	      --movie_kb_path .\deep_dialog\data\movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 500
	      --simulation_epoch_size 100
	      --write_model_dir .\deep_dialog\checkpoints\rl_agent\
	      --run_mode 3
	      --act_level 0
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path .\deep_dialog\data\user_goals_first_turn_template.part.movie.v1.p
	      --warm_start 1
	      --warm_start_epochs 120
```
[End2End with NLU and NLG]
```sh
python run.py --agt 9
	      --usr 1
	      --max_turn 40
	      --movie_kb_path .\deep_dialog\data\movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 500
	      --simulation_epoch_size 100
	      --write_model_dir .\deep_dialog\checkpoints\rl_agent\
	      --run_mode 3
	      --act_level 1
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path .\deep_dialog\data\user_goals_first_turn_template.part.movie.v1.p
	      --warm_start 1
	      --warm_start_epochs 120
```
Test RL Agent with N dialogues:
```sh
python run.py --agt 9
	      --usr 1
	      --max_turn 40
	      --movie_kb_path .\deep_dialog\data\movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 300 
	      --simulation_epoch_size 100
	      --write_model_dir .\deep_dialog\checkpoints\rl_agent\
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path .\deep_dialog\data\user_goals_first_turn_template.part.movie.v1.p
	      --trained_model_path .\deep_dialog\checkpoints\rl_agent\noe2e\agt_9_400_420_0.90000.p
	      --run_mode 3
```

## Learning Curve
1. Plotting
``` python draw_learning_curve.py --result_file ./deep_dialog/checkpoints/rl_agent/noe2e/agt_9_performance_records.json```
2. Pull out the numbers and draw the curves in Excel
