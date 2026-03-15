import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO

# ==========================================
# 0. 全域設定與常數
# ==========================================
os.makedirs("./logs/", exist_ok=True)

skills_list = [
    "probability", "pattern-finding", "area", "equation-solving", "multiplication", "inducing-functions",
    "square-root", "symbolization-articulation", "pythagorean-theorem", "multiplying-decimals",
    "interpreting-linear-equations", "reading-graph", "substitution", "properties-of-geometric-figures", "discount"
]
K = len(skills_list)
D_CONSTANT = 20.0
THETA_INIT = 100.0
MAX_STEPS = 30
MAX_THETA = 200.0
THETA_MASTERY = 150.0
DIFFUSION_RATE = 0.15 # 知識擴散係數

# 知識點擴散 DAG 矩陣
raw_relation = np.zeros((K, K))
raw_relation[7, 12] = 1.0; raw_relation[12, 3] = 1.0; raw_relation[3, 10] = 1.0; raw_relation[11, 10] = 1.0
raw_relation[4, 9] = 1.0;  raw_relation[9, 14] = 1.0; raw_relation[4, 2] = 1.0;  raw_relation[2, 8] = 1.0
raw_relation[6, 8] = 1.0;  raw_relation[13, 8] = 1.0; raw_relation[1, 5] = 1.0;  raw_relation[0, 11] = 1.0
relation_matrix = np.zeros((K, K))
for i in range(K):
    out_degree = np.sum(raw_relation[i])
    for j in range(K):
        if i == j: relation_matrix[i, j] = 1.0
        elif raw_relation[i, j] > 0: relation_matrix[i, j] = raw_relation[i, j] / out_degree

# ==========================================
# 1. 讀取真實資料與 MIRT 參數校準
# ==========================================
print("📥 [Phase 1] 正在讀取 ASSISTments 2017 資料集...")
try:
    df = pd.read_csv(r'C:\實驗一\anonymized_full_release_competition_dataset.csv', usecols=['studentId', 'problemId', 'skill', 'correct'])
except FileNotFoundError:
    print("⚠️ 找不到檔案，將生成備用測試用微型資料庫以供除錯...")
    df = pd.DataFrame({
        'studentId': np.random.randint(0, 100, 5000),
        'problemId': np.random.randint(0, 500, 5000),
        'skill': np.random.choice(skills_list, 5000),
        'correct': np.random.randint(0, 2, 5000)
    })

df = df[df['skill'].isin(skills_list)].dropna()
student_ids = df['studentId'].astype('category').cat.codes.values.astype(np.int64)
item_ids = df['problemId'].astype('category').cat.codes.values.astype(np.int64)
skill_ids = df['skill'].astype('category').cat.set_categories(skills_list).cat.codes.values.astype(np.int64)
corrects = df['correct'].values.astype(np.float32)

num_students = len(np.unique(student_ids))
num_items = len(np.unique(item_ids))
item_to_skill = {int(i): int(s) for i, s in zip(item_ids, skill_ids)}
item_skill_tensor = torch.tensor([item_to_skill[i] for i in range(num_items)], dtype=torch.long)

class AssistmentsMIRT(nn.Module):
    def __init__(self, num_students, num_items, item_skill_map):
        super().__init__()
        self.theta = nn.Embedding(num_students, K)
        self.a = nn.Embedding(num_items, 1)
        self.b = nn.Embedding(num_items, 1)
        self.register_buffer('item_skill_map', item_skill_map)
        nn.init.normal_(self.theta.weight, 0, 1)
        nn.init.normal_(self.b.weight, 0, 1)
        nn.init.constant_(self.a.weight, 1.0)

    def forward(self, stu_idx, item_idx):
        skill_idx = self.item_skill_map[item_idx]
        theta_k = self.theta(stu_idx).gather(1, skill_idx.unsqueeze(1)).squeeze()
        a = torch.nn.functional.softplus(self.a(item_idx).squeeze())
        b = self.b(item_idx).squeeze()
        return a * (theta_k - b)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mirt_model = AssistmentsMIRT(num_students, num_items, item_skill_tensor.to(device)).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(mirt_model.parameters(), lr=0.05)
dataset = TensorDataset(torch.tensor(student_ids), torch.tensor(item_ids), torch.tensor(corrects))
dataloader = DataLoader(dataset, batch_size=4096, shuffle=True)

print("🧠 開始 PyTorch MIRT 訓練 (校準學生與題目參數)...")
for epoch in range(5):
    for b_stu, b_item, b_y in dataloader:
        b_stu, b_item, b_y = b_stu.to(device), b_item.to(device), b_y.to(device)
        optimizer.zero_grad()
        loss = criterion(mirt_model(b_stu, b_item), b_y)
        loss.backward()
        optimizer.step()

mirt_model.eval()
with torch.no_grad():
    raw_theta = mirt_model.theta.weight.cpu().numpy()
    raw_a = torch.nn.functional.softplus(mirt_model.a.weight).cpu().numpy().flatten()
    raw_b = mirt_model.b.weight.cpu().numpy().flatten()

REAL_STUDENTS = np.clip(raw_theta * 20.0 + 100.0, 0.0, 200.0)
REAL_ITEM_BANK = [{"id": int(i), "skill": int(item_to_skill[i]), "a": float(raw_a[i]), "b": float(raw_b[i] * 20.0 + 100.0)} for i in range(num_items)]

ALL_STUDENTS_SORTED = REAL_STUDENTS[np.argsort(np.mean(REAL_STUDENTS, axis=1))]
TRAIN_STUDENTS = ALL_STUDENTS_SORTED[[i for i in range(len(ALL_STUDENTS_SORTED)) if i % 5 != 0]]
TEST_STUDENTS = ALL_STUDENTS_SORTED[[i for i in range(len(ALL_STUDENTS_SORTED)) if i % 5 == 0]]

# ==========================================
# 2. 科學嚴謹版強化學習環境 (分層抽樣 + 高斯 ZPD)
# ==========================================
def irt_prob_2pl(theta, a, b):
    return 1.0 / (1.0 + np.exp(-a * (theta - b) / D_CONSTANT))

def fisher_information_2pl(theta, a, b):
    P = irt_prob_2pl(theta, a, b)
    return max(1e-8, P * (1.0 - P) * (a / D_CONSTANT)**2)

class EduRLEnv(gym.Env):
    def __init__(self, mode='train', seed=None):
        super(EduRLEnv, self).__init__()
        self.mode = mode
        self.student_pool = TRAIN_STUDENTS if mode == 'train' else TEST_STUDENTS
        self.item_bank = REAL_ITEM_BANK
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(2*K + 2,), dtype=np.float32)
        # K 維技能優先度：PPO 直接對每個技能輸出優先分數，因果鏈更短
        self.action_space = spaces.Box(low=-3.0, high=3.0, shape=(K,), dtype=np.float32)
        if seed is not None: np.random.seed(seed)

    def reset(self, seed=None, specific_student=None, options=None):
        super().reset(seed=seed)
        if specific_student is not None:
            self.true_theta = specific_student.copy()
        else:
            idx = np.random.randint(len(self.student_pool))
            self.true_theta = self.student_pool[idx].copy()

        # 模擬前測結束的狀態
        measurement_error = np.random.normal(0, 5.0, K)
        self.est_theta = np.clip(self.true_theta + measurement_error, 50.0, 180.0)
        self.est_variance = np.full(K, 50.0) 

        self.learning_speed = np.random.uniform(0.7, 1.3)
        self.fail_streak, self.same_skill_streak, self.last_skill, self.step_count = 0, 0, -1, 0
        self.used_items = set()
        return self._get_obs(), {}

    def _get_obs(self):
        return np.concatenate([(self.est_theta - 100.0)/20.0, self.est_variance/400.0,
                               [self.fail_streak/5.0], [self.same_skill_streak/5.0]], dtype=np.float32)

    def step(self, action):
        # action 是 K 維技能優先度分數，直接對應 15 個技能
        # softmax 轉成機率，取 argmax 作為本步目標技能
        skill_priority = np.exp(action - np.max(action))  # 數值穩定的 softmax
        skill_priority /= np.sum(skill_priority)
        target_skill = int(np.argmax(skill_priority))

        available_items = [i for i in self.item_bank if i["id"] not in self.used_items]
        if not available_items:
            return self._get_obs(), 0.0, True, False, {}

        # 先找目標技能的題目；若已全出完則退回全部可用題目
        skill_items = [i for i in available_items if i["skill"] == target_skill]
        candidate_items = skill_items if skill_items else available_items

        # 在候選題中選 ZPD 最佳：P_est 最接近 0.75 的題
        best_item = min(
            candidate_items,
            key=lambda i: abs(irt_prob_2pl(self.est_theta[i["skill"]], i["a"], i["b"]) - 0.75)
        )
        return self.step_direct(best_item)

    def step_direct(self, best_item):
        self.step_count += 1
        self.used_items.add(best_item["id"])
        k_target = best_item["skill"]
        old_true_theta = self.true_theta.copy()  # 僅供 info/評估用，不參與 reward
        old_est_theta = self.est_theta.copy()    # reward 只用這個

        P_true = irt_prob_2pl(self.true_theta[k_target], best_item["a"], best_item["b"])
        u = 1 if np.random.rand() < P_true else 0

        # 高斯分佈 ZPD
        zpd_efficiency = np.exp(-((P_true - 0.75)**2) / (2 * 0.12**2))
        saturation = (MAX_THETA - self.true_theta[k_target]) / (MAX_THETA - 50.0)

        if u == 1:
            base_gain = 3.5 * zpd_efficiency * saturation * self.learning_speed
            for m in range(K):
                if relation_matrix[k_target, m] > 0:
                    gain = base_gain if m == k_target else (base_gain * relation_matrix[k_target, m] * DIFFUSION_RATE)
                    self.true_theta[m] = min(MAX_THETA, self.true_theta[m] + gain)
        else:
            base_gain = 0.6 * zpd_efficiency * saturation * self.learning_speed
            self.true_theta[k_target] = min(MAX_THETA, self.true_theta[k_target] + base_gain)
            
            # 認知超載懲罰 (這是之前曲線下跌的主因)
            if P_true < 0.2:
                self.true_theta[k_target] = max(0.0, self.true_theta[k_target] - 0.2)

        prob_est = irt_prob_2pl(self.est_theta[k_target], best_item["a"], best_item["b"])
        gradient = (best_item["a"] / D_CONSTANT) * (u - prob_est)
        fisher = fisher_information_2pl(self.est_theta[k_target], best_item["a"], best_item["b"])
        self.est_theta[k_target] += np.clip(gradient / (fisher + 1/400.0), -10.0, 10.0)
        
        self.est_variance[k_target] *= (1.0 - fisher * self.est_variance[k_target])
        self.est_variance[k_target] = max(5.0, self.est_variance[k_target])

        self.fail_streak = 0 if u == 1 else self.fail_streak + 1
        self.same_skill_streak = self.same_skill_streak + 1 if k_target == self.last_skill else 0
        self.last_skill = k_target

        # Reward 完全基於可觀測的 est_theta，不偷看 true_theta
        mean_gain = np.mean(self.est_theta) - np.mean(old_est_theta)
        weak_gain = np.min(self.est_theta) - np.min(old_est_theta)
        r_balance = -0.3 * (np.std(self.est_theta) - np.std(old_est_theta))

        reward = (40.0 * mean_gain) + (60.0 * weak_gain) + r_balance - 0.1
        
        done = self.step_count >= MAX_STEPS

        info = {
            "skill": k_target, "difficulty_b": best_item["b"],
            "u": u, "p_true": P_true, 
            "est_theta": self.est_theta[k_target], "delta_diff": self.est_theta[k_target] - best_item["b"],
            "true_theta_min": np.min(self.true_theta),
            "est_rmse": np.sqrt(np.mean((self.true_theta - self.est_theta)**2)),
            "skill_std": np.std(self.true_theta)
        }
        return self._get_obs(), float(reward), done, False, info

# ==========================================
# 3. 測試與歷程追蹤
# ==========================================
def run_trajectory_test(model, env_fn, test_students, max_steps=30, agent_name="RL"):
    all_process_logs = []
    for stu_idx, init_theta in enumerate(test_students):
        env = env_fn()
        env.reset(specific_student=init_theta)

        for step in range(1, max_steps + 1):
            if agent_name == "RL":
                action, _ = model.predict(env._get_obs(), deterministic=True)
                obs, reward, term, trunc, info = env.step(action)
            else:
                # ZPD 對照組：選 P_est 最接近 0.75 的題（目標與 RL 一致，比較才公平）
                available_items = [i for i in env.item_bank if i["id"] not in env.used_items]
                best_item = min(
                    available_items,
                    key=lambda i: abs(irt_prob_2pl(env.est_theta[i["skill"]], i["a"], i["b"]) - 0.75)
                )
                obs, reward, term, trunc, info = env.step_direct(best_item)

            all_process_logs.append({
                "Agent": agent_name, "Student_ID": stu_idx, "Step": step,
                "Item_Skill": info["skill"], "Difficulty_b": round(info["difficulty_b"], 2),
                "Correct": info["u"], "P_True": round(info["p_true"], 3),
                "Est_Theta": round(info["est_theta"], 2), "Delta_Diff": round(info["delta_diff"], 2),
                "Weakest_Skill_Val": round(info["true_theta_min"], 2),
                "Est_RMSE": round(info["est_rmse"], 3), "Skill_Std": round(info["skill_std"], 3)
            })
            if term: break
    return pd.DataFrame(all_process_logs)

# ==========================================
# 4. 主程式執行
# ==========================================
if __name__ == "__main__":
    print("\n🚀 [Phase 2] 正在訓練建立於真實資料上的 RL Meta-Controller (500,000 steps)...")
    print("   [修正版] Action: K維技能優先度 | Reward: est_theta | 對照: ZPD策略")
    env_fn = lambda: EduRLEnv(mode='train')
    
    rl_model = PPO("MlpPolicy", env_fn(), verbose=1, 
                   learning_rate=3e-4, n_steps=2048, batch_size=128, gamma=0.95)
    rl_model.learn(total_timesteps=500000)
    
    print("\n📈 [Phase 3] 正在進行 RL vs ZPD 的分層微觀教學歷程測試...")
    
    # 計算測試集中所有學生的平均能力值
    test_student_means = np.mean(TEST_STUDENTS, axis=1)
    
    # 尋找最接近目標能力值 (60, 100, 140) 的學生索引
    def get_closest_students(target_val, n=10):
        idx = np.argsort(np.abs(test_student_means - target_val))[:n]
        return TEST_STUDENTS[idx]

    # 定義低、中、高三個能力層次的學生群組
    student_groups = {
        "Low_60": get_closest_students(60.0),
        "Mid_100": get_closest_students(100.0),
        "High_140": get_closest_students(140.0)
    }

    eval_env_fn = lambda: EduRLEnv(mode='test')
    
    all_final_dfs = []

    # 針對三個不同的能力層次分別進行測試
    for group_name, students in student_groups.items():
        print(f"\n🔍 正在測試 {group_name} 程度的學生群組...")
        
        rl_log_df = run_trajectory_test(rl_model, eval_env_fn, students, agent_name="RL")
        zpd_log_df = run_trajectory_test(None, eval_env_fn, students, agent_name="ZPD")
        
        # 加入一個欄位標記這是哪個能力層次的學生，方便後續畫圖區分
        rl_log_df["Ability_Level"] = group_name
        zpd_log_df["Ability_Level"] = group_name
        
        combined_df = pd.concat([rl_log_df, zpd_log_df])
        all_final_dfs.append(combined_df)

    # 合併所有階層的結果並存檔
    final_master_df = pd.concat(all_final_dfs)
    final_master_df.to_csv('./logs/RealData_Teaching_Process_(0315第6版_修正).csv', index=False)

    print("🎉 執行完畢！分層微觀歷程檔案已儲存至 ./logs/RealData_Teaching_Process_(0315第6版_修正).csv")
