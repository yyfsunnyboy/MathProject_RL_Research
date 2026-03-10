import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from sklearn.decomposition import FactorAnalysis
from sklearn.cluster import KMeans
from difflib import SequenceMatcher

# 參數設定
N_STUDENTS = 500
K_TARGET = 15  
TEST_LENGTH = 30
SIGMA = 20
D = 20
a_base = 1.0
rhos = [0.0, 0.3, 0.6, 0.9]
MIN_ITEMS = 120
N_ITEMS_PER_SKILL = 40

np.random.seed(42)

# 固定 15 知識點列表 (索引號0-14)
skills_list = [
    "probability", "pattern-finding", "area", "equation-solving", "multiplication", "inducing-functions",
    "square-root", "symbolization-articulation", "pythagorean-theorem", "multiplying-decimals",
    "interpreting-linear-equations", "reading-graph", "substitution", "properties-of-geometric-figures", "discount"
]
K = len(skills_list)

# 關聯矩陣
relation_matrix = np.zeros((K, K))

# 鏈條 A
relation_matrix[7, 12] = 1; relation_matrix[12, 7] = 1
relation_matrix[12, 3] = 1; relation_matrix[3, 12] = 1
relation_matrix[3, 10] = 1; relation_matrix[10, 3] = 1
relation_matrix[11, 10] = 1; relation_matrix[10, 11] = 1

# 鏈條 B
relation_matrix[4, 9] = 1; relation_matrix[9, 4] = 1
relation_matrix[9, 14] = 1; relation_matrix[14, 9] = 1
relation_matrix[4, 2] = 1; relation_matrix[2, 4] = 1
relation_matrix[2, 8] = 1; relation_matrix[8, 2] = 1
relation_matrix[6, 8] = 1; relation_matrix[8, 6] = 1
relation_matrix[13, 8] = 1; relation_matrix[8, 13] = 1

# 鏈條 C
relation_matrix[1, 5] = 1; relation_matrix[5, 1] = 1
relation_matrix[11, 0] = 1; relation_matrix[0, 11] = 1

np.fill_diagonal(relation_matrix, 1)

# =============================
# 階段1: 預處理 ASSISTments 資料
# =============================
assist_df = pd.read_csv('assistments_full (1).csv', low_memory=False)  # 調整您的檔案路徑

print("ASSISTments columns:", assist_df.columns.tolist())

# 過濾活躍學生
user_interactions = assist_df['studentId'].value_counts()
active_users = user_interactions[user_interactions > 20].index
selected_students = np.random.choice(active_users, min(N_STUDENTS, len(active_users)), replace=False)
assist_logs = assist_df[assist_df['studentId'].isin(selected_students)]

# 產生回應矩陣
response_matrix = assist_logs.pivot_table(
    index='studentId',
    columns='problemId',
    values='correct',
    aggfunc='mean'
).fillna(0).astype(float)

# 知識點映射
item_to_skill = assist_logs[['problemId', 'skill']].drop_duplicates().set_index('problemId')['skill']
item_to_skill = item_to_skill.fillna('unknown')

unique_skills = item_to_skill.unique()
print(f"原始獨特知識點數 (skill): {len(unique_skills)}")

# 群聚知識點 (文字相似度 + MDS + KMeans)
dist_matrix = np.array([[SequenceMatcher(None, s1, s2).ratio() for s2 in unique_skills] for s1 in unique_skills])

# 強制對稱化
dist_matrix = (dist_matrix + dist_matrix.T) / 2

dissimilarity = 1 - dist_matrix
np.fill_diagonal(dissimilarity, 0)

from sklearn.manifold import MDS
embedding = MDS(n_components=2, dissimilarity='precomputed', random_state=42, n_init=10)
skill_emb = embedding.fit_transform(dissimilarity)

kmeans = KMeans(n_clusters=K_TARGET, random_state=42, n_init=10)
skill_clusters = kmeans.fit_predict(skill_emb)

skill_map = dict(zip(unique_skills, skill_clusters))
item_to_skill = item_to_skill.map(skill_map).fillna(0).astype(int)

K = K_TARGET
print(f"最終使用 K = {K}")

# 存檔（使用 to_dict 存 {problemId: skill}）
item_to_skill_dict = item_to_skill.to_dict()
with open('./processed_item_to_skill.json', 'w') as f:
    json.dump(item_to_skill_dict, f, ensure_ascii=False)

response_matrix.to_csv('./processed_response_matrix.csv')

# =============================
# 階段2: 估計真實參數
# =============================
response_matrix = pd.read_csv('./processed_response_matrix.csv', index_col=0)

fa = FactorAnalysis(n_components=K, random_state=42)
theta_estimates = fa.fit_transform(response_matrix.values)
mu = np.mean(theta_estimates, axis=0) * 20 + 100

cov_matrix = np.cov(theta_estimates.T)
if K == 1:
    Sigma = np.array([[SIGMA**2]])
else:
    trace_val = np.trace(cov_matrix)
    Sigma = cov_matrix * (SIGMA**2 / trace_val if trace_val != 0 else 1)

np.save('./real_mu.npy', mu)
np.save('./real_Sigma.npy', Sigma)

# 難度估計（防 log(0)）
p_values = response_matrix.mean(axis=0)
p_values = p_values.clip(1e-6, 1 - 1e-6)  # 防 0/1
attempt_avg = assist_logs.groupby('problemId')['attemptCount'].mean()
time_avg = assist_logs.groupby('problemId')['timeTaken'].mean()
difficulties = -np.log(p_values / (1 - p_values)) * D + 100
difficulties += attempt_avg * 5 + time_avg * 0.1
difficulties = np.clip(difficulties, 0, 200)  # 限制 b 範圍，避免 inf
difficulties = pd.Series(difficulties, index=response_matrix.columns).fillna(100)

# 產生項目（無多技能強制）
items_real = []
with open('./processed_item_to_skill.json', 'r') as f:
    skill_dict = json.load(f)  # dict {problemId: skill}

problem_ids = response_matrix.columns
for pid in problem_ids:
    pid_str = str(pid)
    main_skill = skill_dict.get(pid_str, 0)
    main_skill = int(main_skill) if isinstance(main_skill, (int, np.integer)) else 0

    # 50% 機率單技能（不強制）
    if np.random.rand() < 0.5:
        num_extra = 0
    else:
        num_extra = np.random.randint(1, 3) if K > 1 else 0

    extra_candidates = np.where(relation_matrix[main_skill] == 1)[0].tolist()
    extra_candidates = [int(x) for x in extra_candidates]  # 轉 Python int
    extra_candidates.remove(main_skill) if main_skill in extra_candidates else None

    extra_skills = np.random.choice(extra_candidates or range(K), min(num_extra, len(extra_candidates or range(K))), replace=False) if K > 1 else []
    extra_skills = [int(x) for x in extra_skills]  # 轉 Python int

    skills = [int(main_skill)] + extra_skills  # 確保 int
    a_loadings = np.random.uniform(0.5, 1.5, len(skills))
    a_loadings = [float(x) for x in a_loadings]  # 轉 Python float

    b = float(difficulties.get(pid, 100))

    items_real.append({"skills": skills, "a": a_loadings, "difficulty": b})

with open('./real_items.json', 'w') as f:
    json.dump(items_real, f, ensure_ascii=False, indent=2)

print(f"已產生 {len(items_real)} 個真實項目")

# =============================
# 生成能力（限相關連動）
# =============================
def generate_abilities(rho_override=None):
    mu = np.load('./real_mu.npy')
    Sigma_base = np.load('./real_Sigma.npy')
    if rho_override is not None:
        diag = np.diag(Sigma_base)
        off_diag = rho_override * relation_matrix * np.sqrt(np.outer(diag, diag))
        np.fill_diagonal(off_diag, diag)
        Sigma = off_diag
    else:
        Sigma = Sigma_base
    return np.random.multivariate_normal(mu, Sigma, N_STUDENTS), mu, Sigma

# =============================
# 題庫生成（補充合成）
# =============================
def generate_items():
    with open('./real_items.json', 'r') as f:
        items = json.load(f)
    if len(items) < MIN_ITEMS:
        print(f"補充合成項目...")
        for s in range(K):
            for _ in range(N_ITEMS_PER_SKILL):
                b = np.random.uniform(60, 140)
                num_extra = np.random.randint(0, 3) if np.random.rand() < 0.5 else 0
                extra_candidates = np.where(relation_matrix[s] == 1)[0].tolist()
                extra_candidates.remove(s) if s in extra_candidates else None
                extra_skills = np.random.choice(extra_candidates or range(K), min(num_extra, len(extra_candidates or range(K))), replace=False) if K > 1 else []
                extra_skills = [int(x) for x in extra_skills]
                skills = [int(s)] + extra_skills
                a_loadings = np.random.uniform(0.5, 1.5, len(skills))
                a_loadings = [float(x) for x in a_loadings]
                items.append({"skills": skills, "a": a_loadings, "difficulty": b})
    return items

# =============================
# IRT 與 Fisher
# =============================
def irt_prob(theta, item):
    skills = item["skills"]
    a = np.array(item["a"])
    b = item["difficulty"]
    dot = np.sum(a * (theta[skills] - b))
    return 1 / (1 + np.exp(-dot / D))

def fisher(theta, item):
    P = irt_prob(theta, item)
    a = np.array(item["a"])
    return P * (1 - P) * (np.sum(a)**2 / D**2)

# =============================
# MAP 更新
# =============================
def map_update(theta, responses, used_items, mu, Sigma_inv):
    g = -Sigma_inv @ (theta - mu)
    H = Sigma_inv.copy()
    for u, item in zip(responses, used_items):
        P = irt_prob(theta, item)
        a = np.array(item["a"])
        skills = item["skills"]
        grad = (u - P) * a / D
        outer = np.outer(grad, grad) * P * (1 - P)
        for i, s1 in enumerate(skills):
            g[s1] += grad[i]
            for j, s2 in enumerate(skills):
                H[s1, s2] += outer[i, j]
    theta_new = theta + np.linalg.solve(H, g)
    return theta_new, H

# =============================
# Selection
# =============================
def select_item(strategy, theta, H, items):
    infos = np.array([fisher(theta, item) for item in items])

    if strategy == "baseline":
        return items[np.argmax(infos)]

    elif strategy == "variance":
        posterior_cov = np.linalg.inv(H)
        variances = np.diag(posterior_cov)
        s_star = np.argmax(variances)
        candidates = [item for item in items if s_star in item["skills"]]
        if not candidates:
            return items[np.argmax(infos)]
        candidate_infos = [fisher(theta, c) for c in candidates]
        return candidates[np.argmax(candidate_infos)]

    elif strategy == "trace":
        posterior_cov = np.linalg.inv(H)
        best = -1
        best_item = None
        for item in items:
            I = fisher(theta, item)
            skills = item["skills"]
            avg_var = np.mean([posterior_cov[s, s] for s in skills])
            delta = avg_var * I
            if delta > best:
                best = delta
                best_item = item
        return best_item or items[np.argmax(infos)]

# =============================
# 主模擬
# =============================
def run_sim(strategy, rho_override=None):
    true_theta, mu, Sigma = generate_abilities(rho_override=rho_override)
    Sigma_inv = np.linalg.inv(Sigma)
    items = generate_items()

    mae_curve = np.zeros(TEST_LENGTH)
    rmse_curve = np.zeros(TEST_LENGTH)
    bias_curve = np.zeros(TEST_LENGTH)
    var_curve = np.zeros(TEST_LENGTH)
    trace_curve = np.zeros(TEST_LENGTH)

    for i in range(N_STUDENTS):
        theta = np.ones(K) * 100
        responses = []
        used_items = []
        H = Sigma_inv.copy()

        for t in range(TEST_LENGTH):
            item = select_item(strategy, theta, H, items)
            P_true = irt_prob(true_theta[i], item)
            u = np.random.binomial(1, P_true)

            responses.append(u)
            used_items.append(item)

            theta, H = map_update(theta, responses, used_items, mu, Sigma_inv)

            error = theta - true_theta[i]
            mae_curve[t] += np.mean(np.abs(error))
            rmse_curve[t] += np.sqrt(np.mean(error**2))
            bias_curve[t] += np.mean(error)
            var_curve[t] += np.var(theta)
            trace_curve[t] += np.trace(np.linalg.inv(H))

    mae_curve /= N_STUDENTS
    rmse_curve /= N_STUDENTS
    bias_curve /= N_STUDENTS
    var_curve /= N_STUDENTS
    trace_curve /= N_STUDENTS

    return mae_curve, rmse_curve, bias_curve, var_curve, trace_curve

# =============================
# 執行比較
# =============================
strategies = ["baseline", "variance", "trace"]
results = {r: {} for r in rhos}

for r in rhos:
    print(f"\n=== 測試 rho = {r} ===")
    for strat in strategies:
        print(f"  運行 {strat} ...")
        results[r][strat] = run_sim(strat, rho_override=r)

labels = ["MAE", "RMSE", "Bias", "Variance", "Trace"]
for i, label in enumerate(labels):
    plt.figure(figsize=(10, 6))
    for r in rhos:
        plt.plot(results[r]["baseline"][i], label=f"Baseline (rho={r})", linestyle='-')
        plt.plot(results[r]["variance"][i], label=f"Variance (rho={r})", linestyle='--')
        plt.plot(results[r]["trace"][i], label=f"Trace (rho={r})", linestyle=':')
    plt.legend()
    plt.title(f"{label} 曲線 (多技能模式)")
    plt.xlabel("題目數")
    plt.ylabel(label)
    plt.grid(True)
    plt.show()

print("\n最終 MAE 比較表（多技能模式）：")
print("rho\tBaseline\tVariance\tTrace")
for r in rhos:
    mae_b = results[r]["baseline"][0][-1]
    mae_v = results[r]["variance"][0][-1]
    mae_t = results[r]["trace"][0][-1]
    print(f"{r}\t{mae_b:.4f}\t{mae_v:.4f}\t{mae_t:.4f}")
