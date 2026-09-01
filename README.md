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
This notebook takes user inputted strategies which are then used to simulate races using a Monte Carlo model. The model predicts lap times using a random forest regression which was created with data from the FastF1 API from Barcelona '26 which is combined with a tyre cliff function as the data didn't include tyre cliffs therefore the regression model does not account for tyre cliffs. Race flag transitions are controlled by a markov chain to give random flags for each race simulated. 

To use the notebook begin by running each cell. Key parameters are set in the cell after the random forest regression. Key parameters include the number of laps for each race, the number of simulations ie races simulated and finally the strategies that are tested. By default there are 5 strategies pre entered, the first 3 were self created, strategy 4 is Hamilton's strategy from the race with the final pit lap being moved back a lap (assuming the VSC didn't occur) and the 5th strategy is George's 2 stop. To enter your own strategy either edit a pre existing strategy or create a new nested list with same structure as the others. Ensure the first nest list begins with -1 to define the starting tyre with the option of "SOFT" "MEDIUM" "HARD".

Continue to run the following cells. The second to last cell runs the Monte Carlo simulation. Depending on the number of simulations chosen and the hardware of the device this can take upwards oof 10 minutes, however once the simualtions have been completed "Finished" will be printed. The final cell will then output the average race time of each strategy. When simulating multiple strategies over N simulations the program will pre generate N "races" of flags so each strategy will experience the same flags to ensure fair comparison.

## Reinforcement Learning Model
This part of the project includes a reinforcement learning agent that learns optimal strategy decisions. A custom gymnasium environment models the race including flag states, tyre wear and lap times from the deterministic strategy. The agent is trained using proximal policy optimization (PPO) from Stable-Baselines3. The trained model can then be ran on race simulations outputting the decisions the model made and the laptimes.

### Training
Although the repository includes a pre‑trained agent (RL_model_final.zip), users can train new agents using the provided notebooks. Training can be performed using either a single environment (RL_training.ipynb) or parallel environments (RL_parallel_processing_training.ipynb), with the parallel version offering significantly faster training.

On my laptop, training for 500,000 steps took upwards of 3 hours using a single environment, compared to 40 minutes using parallel environments. Performance depends on hardware and training configuration.

#### Single Environment Training Open RL_training.ipynb.

1. Open "RL_training.ipynb"

2. Run all setup cells before the final training cell.

3. In the final cell:
- Set a model name for saving the trained agent.
- Adjust training parameters if desired.

4. Key parameters:
- ent_coef — controls exploration (higher values increase exploration).
- training_steps — more steps generally improve performance but increase training time.

5. The Gymnasium environment is defined earlier in the notebook; modifying observations or reward structure will affect training behaviour.

6. During training, several statistics are printed:
- training_steps
- ep_rew_mean (×10 ≈ average race time)
- entropy_loss (lower values indicate reduced exploration)
- explained_variance (should converge to 0.995–0.999 for a correctly functioning model)

A well‑trained model typically shows:
explained variance > 0.995
entropy loss < –0.23
ep_rew_mean between 56.5 and 57

If explained variance does not converge, there is likely an issue in the environment logic.

#### Parallel Environment Training 
Parallel training uses vectorised environments to significantly reduce training time. The environment is defined in "race_env.py", and training is executed in "RL_parallel_processing_training.ipynb".

The setup mirrors the single‑environment version, with one additional parameter:

n_envs — number of parallel environments (I used 10 on a 16‑core CPU).

Training statistics are printed in the same format as the single‑environment version.
Checkpoints are automatically saved every 50,000 steps to protect against crashes.

### Evaluation and Implementation of Trained Agents
Evaluation is performed using RL_simulation_and_visualisation.ipynb. This notebook allows users to load any trained agent and run race simulations. Two evaluation modes are available:

- Single race simulation — runs one full race and visualises the agent’s strategy.

- Multiple race simulation — runs N races and outputs each race time and the average race time.

#### Single Race Simulation
1. Run the first cell to import libraries and load the environment.
2. Enter the name of the agent to simulate (found in the reinforcement_learning folder).

3. The next cell simulates one full race:
- Flag states are generated each lap.
- The agent chooses between staying out or pitting for soft, medium, or hard tyres.
- A CSV file (strategy_output.csv, name can be changed) is generated containing:
    - the agent’s decision each lap
    - the flag state
    - the lap time
    - total race time

A plot is generated showing lap times, with pit stops marked using dashed lines.

#### Multiple Race Simulations
The final cell allows the user to set n_simulations.
The notebook then runs that number of races using the loaded agent and outputs:

- each race’s total time
- the average race time across all simulations

## Results
Deterministic strategy - The main goal of the project was to determine whether the 2 stop or 3 stop was faster. Using the deterministic model I found the 3 stop's race time was 13.26 seconds faster than the 2 stop over 1000 simulations. Obviously there is some nuance to this result as the model doesn't account for time lost to traffic, dirty air and having to overtake other drives which would obviously likely hinder the 3 stop more than the 2 stop. The laptime model also doesn't account for driver, team or driver behaviour (for example saving tyres). However, I think the time difference shown by the model clearly implies the 3 stop was the faster strategy. Furthermore, after running 100 simulations with set flags that didn't include the late VSC that gave Hamilton the free pitstop in the actual race and once again the results implied the 3 stop was quicker. 

## Future Work
While the project did achieve what I wanted it to there are multiple improvements that could be made including:
- Adding wet weather simulation
- Improving the RL agent
- Introducing other cars to race against
- Improve laptime predictions to account for dirty air, slipstream and battling

## Libraries used
NumPy
Pandas
Scikit-Learn
Gymnasium
Stable-Baselines3
FastF1
Matplotlib

## Acknowledgements 
This project uses Stable-Baselines3 (Raffin et al., 2021) for PPO training.
https://github.com/DLR-RM/stable-baselines3

Barcelona '26 data sourced from FastF1 API
