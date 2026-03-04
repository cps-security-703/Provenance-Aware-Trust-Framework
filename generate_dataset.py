import pandas as pd
import numpy as np
import random

# --- Configuration ---
TARGET_ROWS = 10000 # Target size > 5000
CHANNELS = [
    "OTA",
    "P2P",
    "charging station",
    "nearby car",
    "parking garage",
    "police station",
    "traffic sight",
    "weather station",
    "flood alert",
]
ROLES = ["honest", "adversary"]
CREATORS = ["OEM", "FakeOEM", "TrafficMgmt", "WeatherSvc", "PoliceDept"]
NODE_COUNT = 60
# --- Column Definitions and Value Ranges (Inferred from original data) ---
# t: time/event ID, integer
# node: vehicle node ID, string (e.g., N00 to N59)
# role: source role, string (honest/adversary)
# channel: communication channel, string
# path_len: integer, path length
# creator: source of the update, string
# ver: version, integer
# is_malicious: 0 or 1
# baseline_accept: 0 or 1
# llm_score: float (0.0 to 1.0)
# llm_accept: 0 or 1
# authentic: 0 or 1
# rollforward: 0 or 1
# chain_ok: 0 or 1

def generate_row(idx):
    """Generates a single row of data."""
    # Base values
    role = random.choice(ROLES)
    channel = random.choice(CHANNELS)
    creator = random.choice(CREATORS)
    is_malicious = 1 if role == "adversary" else 0
    t = random.randint(1, 500)
    node = f"N{random.randint(0, NODE_COUNT-1):02d}"
    ver = random.randint(1, 30)
    path_len = random.randint(2, 8) if channel in ["P2P", "nearby car"] else 2 # OTA and infrastructure channels likely have short path_len

    # Derived values based on role and channel
    # Simulate the outcome columns (baseline_accept, llm_score, llm_accept, authentic, rollforward, chain_ok)
    
    if role == "honest":
        # Honest sources are mostly accepted, but can sometimes be rejected (e.g., due to network error, minor issue)
        authentic = 1
        
        # Determine acceptance probabilities based on channel trust
        if channel in ["OTA", "P2P"]:
            # High trust channels
            p_accept = 0.95
        elif channel in ["charging station", "traffic sight", "weather station", "flood alert"]:
            # Medium trust infrastructure
            p_accept = 0.85
        else:
            # Lower trust P2P/nearby sources
            p_accept = 0.75
            
        baseline_accept = np.random.choice([1, 0], p=[p_accept, 1-p_accept])
        
        # LLM score simulation: honest sources should score high
        llm_score = np.clip(np.random.normal(loc=0.85, scale=0.15), 0.0, 1.0)
        llm_accept = 1 if llm_score > 0.5 else 0
        rollforward = 1
        chain_ok = 1
        
        # Introduce a small chance of failure for honest sources (e.g., bad data, network issue)
        if random.random() < 0.05:
            # Simulate a scenario where an honest source is mistakenly flagged or fails a check
            baseline_accept = 0
            llm_accept = 0
            llm_score = np.clip(np.random.normal(loc=0.3, scale=0.2), 0.0, 0.5)
            authentic = 0 # Authentic data source, but data itself might be corrupted/unauthentic in this instance
            chain_ok = np.random.choice([0, 1])
            
    else: # role == "adversary"
        # Adversarial sources are mostly rejected, but can sometimes be accepted (false negative)
        authentic = 0
        
        # Determine false acceptance probabilities based on channel security
        if channel in ["OTA", "police station"]:
            # High security channels, low false acceptance risk
            p_false_accept = 0.05
        elif channel in ["P2P", "charging station", "traffic sight"]:
            # Medium security channels
            p_false_accept = 0.15
        else:
            # Low security channels (e.g., nearby car, parking garage)
            p_false_accept = 0.25
            
        baseline_accept = np.random.choice([0, 1], p=[1-p_false_accept, p_false_accept])
        
        # LLM score simulation: adversarial sources should score low
        llm_score = np.clip(np.random.normal(loc=0.2, scale=0.15), 0.0, 1.0)
        llm_accept = 1 if llm_score > 0.5 else 0
        rollforward = np.random.choice([0, 1])
        chain_ok = np.random.choice([0, 1])
        
        # If accepted by the baseline, simulate a successful attack with high scores
        if baseline_accept == 1:
            llm_score = np.clip(np.random.normal(loc=0.7, scale=0.2), 0.5, 1.0)
            llm_accept = 1
            rollforward = 1
            chain_ok = 1
        
        # If LLM rejects, ensure score is low
        if llm_accept == 0 and llm_score > 0.5:
             llm_score = np.clip(np.random.normal(loc=0.2, scale=0.1), 0.0, 0.5)

    # Final consistency checks
    if chain_ok == 0:
        rollforward = 0
        
    if rollforward == 0:
        baseline_accept = 0
        llm_accept = 0
        
    if authentic == 0 and creator == "OEM":
        # Simulate a scenario where an adversary spoofs the OEM
        creator = "OEM"
    elif authentic == 1 and creator == "FakeOEM":
        # Honest source should not be FakeOEM, correct this
        creator = random.choice(["OEM", "TrafficMgmt", "WeatherSvc", "PoliceDept"])


    return {
        "t": t,
        "node": node,
        "role": role,
        "channel": channel,
        "path_len": path_len,
        "creator": creator,
        "ver": ver,
        "is_malicious": is_malicious,
        "baseline_accept": baseline_accept,
        "llm_score": round(llm_score, 2),
        "llm_accept": llm_accept,
        "authentic": authentic,
        "rollforward": rollforward,
        "chain_ok": chain_ok,
    }

# Generate the data
data = [generate_row(i) for i in range(TARGET_ROWS)]
df = pd.DataFrame(data)

# Save the expanded dataset
output_filename = "./data/expanded_update_simulation_logs.csv"
df.to_csv(output_filename, index=False)

print(f"Successfully generated dataset with {len(df)} rows to {output_filename}")
