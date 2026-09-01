# f1-strategy-ML
Race strategy simulation using Barcelona ’26 data, including a user inputted strategy tester and a custom Gymnasium environment for training a PPO agent with Stable‑Baselines3.

## Project Overview
This project models Formula 1 race strategy using Barcelona ’26 data. The initial motivation was to analyse Lewis's 3 stop against George's 2 stop, particularly in scenarios where the late Virtual Safety Car doesn't occur. Barcelona was selected because it is considered a balanced “all‑rounder” circuit and is widely used as a test track, making it ideal for modelling tyre wear, stint length, lap‑time evolution, and safety‑car dynamics.

The project also served as a way to deepen my practical experience with NumPy, pandas, and machine learning techniques in Python. Data was collected using the FastF1 API, cleaned with pandas, and used to train a random forest regression model with Scikit‑Learn to predict lap times. NumPy was used throughout and visualisations were created using Matplotlib.

The repository is organised into two sub‑projects:

- Deterministic Strategy Simulator — allows user‑defined strategies to be evaluated using tyre‑wear modelling, lap‑time regression, and safety‑car probability.

- Reinforcement‑Learning Agent — a custom Gymnasium environment where a PPO agent (Stable‑Baselines3) learns optimal pit‑stop and tyre‑compound strategies under stochastic race conditions.

## Repository Structure 
```
f1-strategy-ML/
│
├── deterministic_strategy/
│   - deterministic_race_strategy.ipynb
│     The notebook simulates user inputted strategies
│
├── reinforcement_learning/
│   Contains the reinforcement‑learning implementation:
│   - race_env.py
│     Creates the gym environment and is used for parallel environment training
│   - RL_training.ipynb
│     Notebook for training the PPO agent using Stable‑Baselines3.
│   - RL_parallel_processing_training.ipynb
│     Optional notebook for faster PPO training using parallel environment training.
│   - RL_simulation_and_visualisation.ipynb
│     Notebook for evaluating the trained agent and visualising race outcomes.
│   - RL_model_final.zip
│     A trained PPO model using the hyperparameters in race_env.py and the RL_parallel_processing_training.ipynb
│   - strategy_output.csv
│     The strategy output when running RL_simulation_and_visualisation.ipynb
│
├── checkpoints/
│   Stores additional PPO model checkpoints and logs.
│
├── __pycache__/
│   Auto‑generated Python cache files.
│
├── requirements.txt
│   Python dependencies required to run the project.
│
├── README.md
│   Project documentation and overview.
│
├── LICENSE
│   MIT license for open‑source use.
│
└── .gitignore
    Specifies files and folders Git should ignore
```

## Determinisitc Strategy Simulator
This notebook takes user inputted strategies which are then used to simulate races using a Monte Carlo model. The model predicts lap times using a random forest regression which was created with data from the FastF1 API from Barcelona '26 which is combined with a tyre cliff function. Race flag transitions are controlled by a markov chain to give random flags for each race simulated. 

To use the notebook begin by running each cell. Key parameters are set in the cell after the random forest regression. Key parameters include the number of laps for each race, the number of simulations ie races simulated and finally the strategies that are tested. By default there are 5 strategies pre entered, the first 3 were self created, strategy 4 is Hamilton's strategy from the race with the final pit lap being moved back a lap (assuming the VSC didn't occur) and the 5th strategy is George's 2 stop. To enter your own strategy either edit a pre existing strategy or create a new nested list with same structure as the others. Ensure the first nest list begins with -1 to define the starting tyre with the option of "SOFT" "MEDIUM" "HARD".

Continue to run the following cells. The second to last cell runs the Monte Carlo simulation. Depending on the number of simulations chosen and the hardware of the device this can take upwards oof 10 minutes, however once the simualtions have been completed "Finished" will be printed. The final cell will then output the average race time of each strategy. When simulating multiple strategies over N simulations the program will pre generate N "races" of flags so each strategy will experience the same flags to ensure fair comparison.

## Reinforcement Learning Model
This part of the project includes a reinforcement learning agent that learns optimal strategy decisions. A custom gymnasium environment models the race including flag states, tyre wear and lap times. The agent is trained using proximal policy optimization (PPO) from Stable-Baselines3. The trained model can then be ran on race simulations outputting the decisions the model made and the laptimes.

### Training
While the project does already have a trained agent under "RL_model_final.zip" it is also possible to train agents using the environment. An agent can either be trained under 1 environment using the "RL_training.ipynb" or using parallel environments using race_env.py to create the environment and "RL_parallel_processing_training.ipynb" to execute the parallel environment training which I found to be 4x faster, also dependent on training steps and hardware. Training using 500,000 training steps on my laptop in the single environment took upwards of 3 hours which reduced to 40 minutes with the implementation of parallel environments.

Single Environment Training - Begin by opening "RL_training.ipynb" and begin by running all the cells before the final cell. The final cell is responsible for the training. Before starting training ensure to set a name for the model to save under at the end of the cell to allow it to be loaded for simulations. The training parameters can also be found in the same cell. The main parameters to adjust are the ent_coef which impacts how much the model will explore in training (a higher coefficient leads to more exploration) and also training_steps with more training steps commonly leading to better performance from the agent, however training will take longer with more training steps. The gymnasium environment can also be found before the training and adjusting parameters such as the observations and the reward structure can also impact training. While training the model, training statistics will be printed. The main statistics to keep an eye on are the training_steps, ep_rew_mean (x10 will give the models current average race time), entropy_loss (how much the model is still exploring) and explained variance which should reach between 0.999 and 0.995 for a correctly trained model. If explained variance doesn't converge there is likely a mistake in the gymnasium environment. In my experience I found a well trained model had an explained variance >0.995, an entropy loss of < -0.23 and a rew_mean between 56.5 and 57.

Parallel Environment Training - The model can also be trained using parallel environments which can significantly decrease training time. Parallel training is similar to single environment training however the gymnasium environment is stored within race_env.py. To begin the training open the "RL_parallel_processing_training.ipynb". The parameter setup is identical to the single environment version in terms of setting training steps, training parameters and the name the trained agent is saved under. With the addition of parallel environments the number of parallel environments also needs to be set under n_envs towards the top of the cell. I used 10 environments as my laptop has 16 CPU cores. Once again while the agent is training the training statistics are printed. For both versions of training checkpoints are also created every 50,000 training steps in case of program or hardware crashes. 

### Evaluation and Implementation of Trained Agents
Once the agent is trained and saved open the "RL_simulation_and_visualisation.ipynb" notebook. This notebook allows the user to choose which agent they would like to simulate. 2 simulation options are then available:
- Simulate 1 race allowing the loaded agent to decide the strategy which can then be visualised
- Simulate N races and output each races time and average race time

After opening the notebook begin by running the first cell to import the required libraries and the gymnasium environment. In the cell below enter the name of the agent you would like to simulate, these can be found in the RL_strategy folder and can either be the pre-trained model or a model you have trained and saved yourself. The next cell simulates 1 race, where the flags are simulated each lap and the trained agent makes a decision each lap from Staying out or pitting for either soft, medium or hard tyres. This cell will also output a csv called strategy_output (the name can be changed for comparing multiple races or agents). The csv record the decision the agent makes each lap, the flag each lap, and the laptime each lap and total laptimes. The cell below the creates a graph showing the laptime each lap and pitstop laps are marked with a dashed line.
The cell below allows the user to choose how many simulations they would like to run under n_simulations. These simulations are then run with the loaded agent making the strategy decisions in each simualtion. The total race time of each simulation and the average race time is the outputted.

## Acknowledgements 
This project uses Stable-Baselines3 (Raffin et al., 2021) for PPO training.
https://github.com/DLR-RM/stable-baselines3

Barcelona '26 data sourced from FastF1 API
