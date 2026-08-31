# f1-strategy-ML
Race strategy simulation using Barcelona ’26 data, including a user inputted strategy tester and a custom Gymnasium environment for training a PPO agent with Stable‑Baselines3.

## Project Overview
I began the project to deepen my understanding and application of Python libraries and machine learning. I chose an F1 based project as I have been a long time fan so also have a deep understanding of the sport. Barcelona 2026 was chosen as it's seen as a classic "all-rounder" track and after watching the race I wanted to compare Lewis' 3 stop strategy to George's 2 stop strategy particularly in the scenario where the late vsc doesn't occur.

The project includes 2 sub-projects: a deterministic strategy simulator allowing for user inputted strategies and a reinforcement-learning agent trained in a gymnasium environment.

Data was collected from the FastF1 API, cleaned and filtered using Pandas which was used to create a random forest regression model with Scikit-Learn. The gymnasium environment was created using the gymnasium library and the agent was trained using PPO from Stable-Baselines 3. NumPy was used throughout and visualisations were created using Matplotlib.



## Achknowledgements
This project uses Stable-Baselines3 (Raffin et al., 2021) for PPO training.
https://github.com/DLR-RM/stable-baselines3

Barcelona '26 data sourced from FastF1 API
