import torch
import torch.nn as nn
import json
import joblib
import numpy as np

# ------------------------------
# Encoder Model
# ------------------------------

class Encoder(nn.Module):
    def __init__(self, input_dim=5, hidden_dim=16, latent_dim=8):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(self, x):
        return self.network(x)

# ------------------------------
# MultiTask Model
# ------------------------------

class MultiTaskModel(nn.Module):
    def __init__(self, encoder):
        super(MultiTaskModel, self).__init__()
        self.encoder = encoder
        
        self.heads = nn.ModuleDict({
            "predict_math_score": nn.Linear(8, 1),
            "predict_pass_fail": nn.Linear(8, 1),
            "predict_reading_score": nn.Linear(8, 1),
            "predict_writing_score": nn.Linear(8, 1)
        })
    
    def forward(self, x):
        Z = self.encoder(x)
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(Z)
        return outputs

# ------------------------------
# Load Everything
# ------------------------------

scaler = joblib.load("scaler.pkl")

with open("goals_config.json", "r") as f:
    selected_goal_names = json.load(f)

encoder = Encoder()
encoder.load_state_dict(torch.load("encoder.pth"))

model = MultiTaskModel(encoder)
model.load_state_dict(torch.load("multitask_model.pth"))

model.eval()

# ------------------------------
# Gating Function
# ------------------------------

def gating_function(x_input):
    with torch.no_grad():
        Z = model.encoder(x_input)
        z_mean = Z.mean().item()
        
        active_goals = []
        
        if z_mean < 0:
            for g in ["predict_pass_fail", "predict_math_score"]:
                if g in selected_goal_names:
                    active_goals.append(g)
        else:
            for g in ["predict_reading_score", "predict_writing_score"]:
                if g in selected_goal_names:
                    active_goals.append(g)
        
        if len(active_goals) == 0:
            active_goals = selected_goal_names[:1]
        
        return active_goals

# ------------------------------
# Inference Function
# ------------------------------

def predict(input_list):
    x = np.array(input_list).reshape(1, -1)
    x = scaler.transform(x)
    x_tensor = torch.tensor(x, dtype=torch.float32)
    
    outputs = model(x_tensor)
    active_goals = gating_function(x_tensor)
    
    results = {}
    
    for goal in active_goals:
        pred = outputs[goal]
        
        if "pass_fail" in goal:
            prob = torch.sigmoid(pred).item()
            results[goal] = "Pass" if prob > 0.5 else "Fail"
        else:
            results[goal] = float(pred.item())
    
    return results


# ------------------------------
# TEST RUN (IMPORTANT)
# ------------------------------

if __name__ == "__main__":
    sample = [0, 1, 2, 1, 0]  # example input
    print(predict(sample))