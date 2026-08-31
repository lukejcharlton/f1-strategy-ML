# f1-strategy-ML
Race strategy simulation using Barcelona ’26 data, including a user inputted strategy tester and a custom Gymnasium environment for training a PPO agent with Stable‑Baselines3.

## Project Overview
This project models Formula 1 race strategy using Barcelona ’26 data. The initial motivation was to analyse Lewis's 3 stop against George's 2 stop, particularly in scenarios where the late Virtual Safety Car doesn't occur. Barcelona was selected because it is considered a balanced “all‑rounder” circuit and is widely used as a test track, making it ideal for modelling tyre wear, stint length, lap‑time evolution, and safety‑car dynamics.

The project also served as a way to deepen my practical experience with NumPy, pandas, and machine learning techniques in Python. Data was collected using the FastF1 API, cleaned with pandas, and used to train a random forest regression model with Scikit‑Learn to predict lap times. NumPy was used throughout and visualisations were created using Matplotlib.

The repository is organised into two sub‑projects:

- Deterministic Strategy Simulator — allows user‑defined strategies to be evaluated using tyre‑wear modelling, lap‑time regression, and safety‑car probability.

- Reinforcement‑Learning Agent — a custom Gymnasium environment where a PPO agent (Stable‑Baselines3) learns optimal pit‑stop and tyre‑compound strategies under stochastic race conditions.



## Acknowledgements 
This project uses Stable-Baselines3 (Raffin et al., 2021) for PPO training.
https://github.com/DLR-RM/stable-baselines3

Barcelona '26 data sourced from FastF1 API
