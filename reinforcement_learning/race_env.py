import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import gymnasium as gym
from gymnasium import spaces
import fastf1

# Loading race data from fastf1 api

race = fastf1.get_session(2026, "Barcelona", "R")
race.load()
laps = race.laps

# Sort data to use to generate model
laps1 = laps[["Team", "Driver", "LapTime", "LapNumber", "PitOutTime", "PitInTime", "Compound", "TyreLife", "TrackStatus", "IsAccurate"]]
laps2 = laps1[(laps1["LapNumber"] == 1.0) | ((laps1["IsAccurate"] == True) & (laps1["TrackStatus"] == "1"))]
laps3 = laps2[laps2["Team"].isin(["McLaren", "Mercedes", "Ferrari", "Red Bull Racing"])]
laps3 = laps3[laps3["Driver"] != "HAD"]
lapsA = laps3[laps3["PitInTime"].isna() & laps3["PitOutTime"].isna()]
lapsA["LapTime"] = lapsA["LapTime"].dt.total_seconds()
compound_dummies = pd.get_dummies(lapsA["Compound"], prefix="Compound")
lapsfinal = pd.concat([lapsA, compound_dummies], axis=1)
lapsfinal


# Generate lap time prediction model using random forest regression

# Define features and target

model_features = ["LapNumber", "TyreLife", "Compound_HARD", "Compound_MEDIUM", "Compound_SOFT"]
X = lapsfinal[model_features].values
y = lapsfinal["LapTime"].values

# define model sets and parameters

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 5)

laptime_model = RandomForestRegressor(
    n_estimators=400,
    max_depth=30,
    min_samples_split=3,
    min_samples_leaf=2,
    random_state=5
)


# Fitting model 

laptime_model.fit(X_train, y_train)


# Testing model and calculating residual standard deviation to add noise

y_pred = laptime_model.predict(X_test)
print(mean_absolute_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
residuals = y_test - y_pred

laptime_std = residuals.std()


# Core parameters

# flag hot encoding 0 = green, 1 = sc, 2 = vsc

flag_hot_encoding = {
    0: [1, 0, 0],
    1: [0, 1, 0],
    2: [0, 0, 1]
}

flag_transition_matrix = np.array([[0.963, 0.03, 0.007],
                            [0.55, 0.45, 0],
                            [0.9, 0, 0.1]])


tyre_compound_encoding = {
                -1: [0, 0, 0],
                "SOFT": [0, 0, 1],
                 "MEDIUM": [0, 1, 0],
                 "HARD": [1, 0, 0]}





# Generating lap times

def tyre_cliff(tyre_life, current_tyre_compound):
    if current_tyre_compound == "SOFT" and tyre_life > 15:
        return (0.15 * (tyre_life - 15)) ** 1.5
        
    if current_tyre_compound == "MEDIUM" and tyre_life > 21:
        return (0.12 * (tyre_life - 21)) ** 1.3
        
    if current_tyre_compound == "HARD" and tyre_life > 25:
        return (0.12 * (tyre_life - 25)) ** 1.2
        
    return 0

# non pit lap 

def race_lap_time(current_flag, prediction_data, current_tyre_compound):
    tyre_life = prediction_data[0][1]
    
    if current_flag == 0:
        laptime = float(laptime_model.predict(prediction_data)[0]) + np.random.normal(0, laptime_std-0.2) + tyre_cliff(tyre_life, current_tyre_compound)
        return laptime
    
    elif current_flag == 1:
        laptime = float((1.56 * laptime_model.predict(prediction_data))[0]) + np.random.normal(0, laptime_std-0.1)
        return laptime
   
    elif current_flag == 2:
        laptime = float((1.33 * laptime_model.predict(prediction_data))[0]) + np.random.normal(0, laptime_std)
        return  laptime

# pit lap

def pit_lap(current_flag, prediction_data, current_tyre_compound):
    tyre_life = prediction_data[0][1]
    
    if current_flag == 0:
        return  float(laptime_model.predict(prediction_data)[0]) + np.random.normal(0, laptime_std) + 22 + np.random.normal(0, 0.7)+tyre_cliff(tyre_life, current_tyre_compound)
    
    elif current_flag == 1:
        return  float((1.56 * laptime_model.predict(prediction_data)[0])) + np.random.normal(0, laptime_std) + 16.3 + np.random.normal(0, 0.7)
   
    elif current_flag == 2:
        return  float((1.33 * laptime_model.predict(prediction_data)[0])) + np.random.normal(0, laptime_std) + 13.5 + np.random.normal(0, 0.7)



# gym environment
class RaceEnv(gym.Env):
    def __init__(self):
        super().__init__()

        self.lap = None
        self.tyre_life = None
        self.compound = None
        self.flag_state = None
        self.stint = None
        self.compounds_used = None

        self.lap_times = []
        self.compound_history = []
        self.flag_history = []
        self.pit_laps = []
        self.total_time = 0.0

        self.action_space = spaces.Discrete(4)  

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(9,),
            dtype=np.float32
        )

    def reset(self, seed=5, options=None):
        super().reset(seed=seed)

        # Reset state
        self.lap = 0
        self.tyre_life = 1
        self.compound = -1
        self.flag_state = 0
        self.stint = 1
        self.compounds_used = set()

        # Reset telemetry
        self.lap_times = []
        self.compound_history = []
        self.flag_history = []
        self.pit_laps = []
        self.total_time = 0.0

        return self._get_obs(), {}

    def _get_obs(self):
        return np.array([
            self.lap / 66,
            min(self.tyre_life / 66, 1.0),
            *tyre_compound_encoding[self.compound],
            *flag_hot_encoding[self.flag_state],
            min(self.stint / 5, 1.0)
        ], dtype=np.float32)

    def decode_compound(self, action):
        if action == 1: return "HARD"
        if action == 2: return "MEDIUM"
        if action == 3: return "SOFT"

    def step(self, action):

                    
        if self.compound == -1:
            
            if action == 0:
                action = np.random.choice([1, 2, 3])
                
            self.compound = self.decode_compound(action)
            self.compounds_used.add(self.compound)
            
            laptime = 0.0           
        
            self.total_time += laptime
            self.lap_times.append(laptime)
            self.compound_history.append(self.compound)
            self.flag_history.append(self.flag_state)

            info = {
            "lap_time": laptime,
            "compound": self.compound,
            "flag": self.flag_state,
            "pitted": False,
            "lap": self.lap,
            "total_time": self.total_time
        }

            self.lap += 1
            self.tyre_life = 1


            return self._get_obs(), 0.0, False, False, info

        if action in [1, 2, 3]:
            
            prediction_data = [[self.lap, self.tyre_life] + tyre_compound_encoding[self.compound]]
            laptime = pit_lap(self.flag_state, prediction_data, self.compound)

            
            self.compound = self.decode_compound(action)
            self.compounds_used.add(self.compound)
            self.stint += 1
            
            self.total_time += laptime
            self.pit_laps.append(self.lap)
            
            self.lap_times.append(laptime)
            self.compound_history.append(self.compound)
            self.flag_history.append(self.flag_state)

            terminated = self.lap >= 66
            reward = -laptime/100

            info = {
            "lap_time": laptime,
            "compound": self.compound,
            "flag": self.flag_state,
            "pitted": True,
            "lap": self.lap,
            "total_time": self.total_time
        }
            
            if terminated and (len(self.compounds_used) < 2 or self.tyre_life <= 6):
                reward -= 10

            self.lap += 1
            self.flag_state = np.random.choice([0, 1, 2], p = flag_transition_matrix[self.flag_state])
            self.tyre_life = 1  
            
            return self._get_obs(), reward, terminated, False, info


        prediction_data = [[self.lap, self.tyre_life] + tyre_compound_encoding[self.compound]]
        laptime = race_lap_time(self.flag_state, prediction_data, self.compound)

        self.total_time += laptime
        self.lap_times.append(laptime)
        self.compound_history.append(self.compound)
        self.flag_history.append(self.flag_state)
        
        terminated = self.lap >= 66
        reward = -laptime/100

        info = {
            "lap_time": laptime,
            "compound": self.compound,
            "flag": self.flag_state,
            "pitted": False,
            "lap": self.lap,
            "total_time": self.total_time
        }
        
        if terminated and (len(self.compounds_used) < 2 or self.tyre_life <= 6):
            reward -= 10

        self.lap += 1
        self.flag_state = np.random.choice([0, 1, 2], p = flag_transition_matrix[self.flag_state])
        self.tyre_life += 1


        return self._get_obs(), reward, terminated, False, info
