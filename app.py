import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate, StratifiedKFold, train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Operation Risk Handling Index",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif; }
.stApp { background: #0a0e1a; color: #e0e6f0; }
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0d1225 0%, #111827 100%); border-right: 1px solid #1e2d4a; }
[data-testid="stSidebar"] * { color: #c9d6ea !important; font-family: 'Syne', sans-serif !important; }
.metric-card { background: linear-gradient(135deg, #131c35 0%, #0f1929 100%); border: 1px solid #1e3050; border-radius: 16px; padding: 24px 28px; margin-bottom: 16px; box-shadow: 0 4px 24px rgba(0,0,0,0.4); transition: transform 0.2s, box-shadow 0.2s; }
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 32px rgba(99,179,255,0.12); }
.metric-card .label { font-size: 13px; letter-spacing: 2px; text-transform: uppercase; color: #5a7aaa; font-family: 'Space Mono', monospace; }
.metric-card .value { font-size: 36px; font-weight: 800; color: #63b3ff; margin-top: 6px; }
.metric-card .sub { font-size: 12px; color: #3d5a80; margin-top: 4px; font-family: 'Space Mono', monospace; }
.section-header { font-size: 28px; font-weight: 800; color: #e8f0fe; letter-spacing: -0.5px; margin-bottom: 4px; }
.section-sub { font-size: 14px; color: #4a6a9a; font-family: 'Space Mono', monospace; letter-spacing: 1px; margin-bottom: 28px; }
.badge-low { background: linear-gradient(135deg, #3d1515, #5c1a1a); color: #ff6b6b; border: 1px solid #7a2020; padding: 8px 20px; border-radius: 50px; font-size: 13px; font-weight: 700; letter-spacing: 1px; font-family: 'Space Mono', monospace; }
.badge-medium { background: linear-gradient(135deg, #3d3015, #5c4a1a); color: #ffd166; border: 1px solid #7a6020; padding: 8px 20px; border-radius: 50px; font-size: 13px; font-weight: 700; letter-spacing: 1px; font-family: 'Space Mono', monospace; }
.badge-high { background: linear-gradient(135deg, #0f3d1f, #155c2e); color: #06d6a0; border: 1px solid #1a7a3e; padding: 8px 20px; border-radius: 50px; font-size: 13px; font-weight: 700; letter-spacing: 1px; font-family: 'Space Mono', monospace; }
.result-box { background: linear-gradient(135deg, #0d1e3a 0%, #091529 100%); border: 1px solid #1e3a6e; border-radius: 20px; padding: 36px; text-align: center; box-shadow: 0 8px 40px rgba(0,100,255,0.1); }
.result-box .risk-level { font-size: 42px; font-weight: 800; margin: 12px 0; }
.result-box .risk-desc { font-size: 15px; color: #6a8db8; line-height: 1.7; margin-top: 12px; }
hr { border: none; border-top: 1px solid #1e2d4a; margin: 28px 0; }
.stButton > button { background: linear-gradient(135deg, #1a3a6e 0%, #0f2a55 100%); color: #63b3ff; border: 1px solid #2a5aaa; border-radius: 12px; font-family: 'Space Mono', monospace; font-size: 14px; letter-spacing: 1px; padding: 12px 32px; transition: all 0.2s; }
.stButton > button:hover { background: linear-gradient(135deg, #2a5aaa 0%, #1a3a6e 100%); box-shadow: 0 4px 20px rgba(99,179,255,0.25); transform: translateY(-1px); }
#MainMenu, Footer{ visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ─── CSV name ──────────────────────────────────────────────────────────────────
import os
CSV_FILE = os.path.join(os.path.dirname(__file__), "responses.csv")

# ─── Load & Preprocess ─────────────────────────────────────────────────────────
@st.cache_data
def load_and_prepare():
    df_raw = pd.read_csv(CSV_FILE)
    df = df_raw.iloc[:, 2:10].copy()
    col_names = [
        "Unclear Instructions", "Deadline Pressure", "Ambiguity Comfort",
        "Dependency Management", "Mistake Handling", "Multi-tasking",
        "Unknown Tech", "Communication"
    ]
    df.columns = col_names

    mappings = [
        {"Wait for someone else to clarify":1,"Ask after some time":2,"Try to figure it out without asking":3,"Ask immediately and proceed":4,"Clarify and suggest an approach":5},
        {"Panic and slow down":1,"Focus only on finishing somehow":2,"Adjust plan partially":3,"Re-plan and communicate clearly":4,"Re-prioritize tasks calmly":5},
        {"Very uncomfortable":1,"Slightly uncomfortable":2,"Neutral":3,"Comfortable":4,"Very comfortable":5},
        {"Wait until others finish":1,"Frequently get stuck":2,"Manage sometimes":3,"Plan around dependencies":4,"Actively coordinate and unblock":5},
        {"Try to hide it":1,"Ignore and move on":2,"Fix silently":3,"Inform and fix":4,"Inform, fix, and prevent recurrence":5},
        {"Miss some tasks":1,"Handle with difficulty":3,"Get confused":2,"Adjust priorities":4,"Confidently re-prioritize":5},
        {"Avoid it":1,"Wait for guidance":2,"Try basic solutions":3,"Learn and attempt":4,"Learn quickly and solve":5},
        {"Don't communicate":1,"Communicate very late":2,"Communicate when asked":3,"Communicate early":4,"Communicate early with solutions":5},
    ]
    for i, m in enumerate(mappings):
        col = df.columns[i]
        df[col] = df[col].map(m)
    df = df.dropna().astype(int).reset_index(drop=True)
    original_count = len(df)

    # Augmentation
    np.random.seed(42)
    augmented = []
    for _ in range(120):
        row = df.sample(1).values[0]
        new_row = [max(1, min(5, int(v + np.random.choice([-2,-1,0,1,2], p=[0.1,0.2,0.4,0.2,0.1])))) for v in row]
        augmented.append(new_row)
    df_final = pd.concat([df, pd.DataFrame(augmented, columns=col_names)], ignore_index=True)
    return df_final, original_count

@st.cache_data
def train_pipeline(_df):
    X = _df.values
    X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_train)
    X_te = scaler.transform(X_test)

    gmm = GaussianMixture(n_components=3, random_state=42, n_init=10)
    gmm.fit(X_tr)

    y_tr = np.argmax(gmm.predict_proba(X_tr), axis=1)
    y_te = np.argmax(gmm.predict_proba(X_te), axis=1)
    mapping = {old: new for new, old in enumerate(np.argsort(gmm.means_.mean(axis=1)))}
    y_tr = np.array([mapping[i] for i in y_tr])
    y_te = np.array([mapping[i] for i in y_te])

    skf     = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    skf_cvs = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    from sklearn.model_selection import cross_val_score

    def make_models():
        return {
            "SVM":                 SVC(C=0.007, kernel='linear', probability=True, random_state=42),
            "Random Forest":       RandomForestClassifier(n_estimators=15, max_depth=2, min_samples_split=15, min_samples_leaf=5, random_state=42),
            "Logistic Regression": LogisticRegression(C=0.005, max_iter=1000),
            "KNN":                 KNeighborsClassifier(n_neighbors=50),
            "Decision Tree":       DecisionTreeClassifier(max_depth=2, random_state=42),
        }

    # --- STEP 1: CV metrics table (fresh models, exactly like notebook cell 15→16) ---
    mdls_cv = make_models()
    for mdl in mdls_cv.values():
        mdl.fit(X_tr, y_tr)

    results = {}
    for name, mdl in mdls_cv.items():
        sc = cross_validate(mdl, X_tr, y_tr, cv=skf,
                            scoring={'accuracy':'accuracy','precision':'precision_weighted',
                                     'recall':'recall_weighted','f1':'f1_weighted'})
        results[name] = {
            'Accuracy':  round(sc['test_accuracy'].mean(),  4),
            'Precision': round(sc['test_precision'].mean(), 4),
            'Recall':    round(sc['test_recall'].mean(),    4),
            'F1 Score':  round(sc['test_f1'].mean(),        4),
        }

    # --- STEP 2: best model via cross_val_score (fresh models, notebook cell 18) ---
    mdls_best = make_models()
    for mdl in mdls_best.values():
        mdl.fit(X_tr, y_tr)

    best_name, best_score_cv = None, 0
    for name, mdl in mdls_best.items():
        score = cross_val_score(mdl, X_tr, y_tr, cv=skf_cvs, scoring='f1_weighted').mean()
        if score > best_score_cv:
            best_score_cv = score
            best_name     = name

    # --- STEP 3: final best model fitted on full X_tr ---
    best_mdl = make_models()[best_name]
    best_mdl.fit(X_tr, y_tr)

    return scaler, gmm, results, (best_name, best_mdl), mapping, X_tr, X_te, y_tr, y_te

# ─── Init ──────────────────────────────────────────────────────────────────────
try:
    df, original_count = load_and_prepare()
except FileNotFoundError:
    st.error(f"❌ CSV file not found: `{CSV_FILE}`\n\nMake sure it is in the same folder as `app.py`.")
    st.stop()

columns = df.columns.tolist()
scaler, gmm, results, best_model, mapping, X_tr, X_te, y_tr, y_te = train_pipeline(df)
labels_map  = {0:"Low Risk Handling", 1:"Medium Risk Handling", 2:"High Risk Handling"}
color_map   = {0:'#ff6b6b', 1:'#ffd166', 2:'#06d6a0'}
badge_class = {0:'badge-low', 1:'badge-medium', 2:'badge-high'}
emoji_map   = {0:'⚠️', 1:'⚡', 2:'✅'}

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='text-align:center; padding:20px 0 10px;'>
        <div style='font-size:40px;'>🛡️</div>
        <div style='font-size:18px; font-weight:800; color:#63b3ff; letter-spacing:1px;'>ORHI</div>
        <div style='font-size:11px; color:#3d5a80; font-family:Space Mono; letter-spacing:2px; margin-top:4px;'>OPERATION RISK HANDLING INDEX</div>
    </div>
    <hr style='border-color:#1e2d4a; margin:16px 0;'>
    """, unsafe_allow_html=True)

    page = st.radio("Nav", ["🏠  Home","📊  Data & Visualizations","🤖  Model Performance","🎯  Risk Predictor"], label_visibility="collapsed")

    st.markdown(f"""
    <hr style='border-color:#1e2d4a; margin:20px 0;'>
    <div style='font-size:11px; color:#2a4060; font-family:Space Mono; text-align:center; line-height:1.8;'>
        REAL RESPONSES: {original_count}<br>AFTER AUGMENT: {len(df)}<br>BEST: {best_model[0][:14].upper()}
    </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ═══════════════════════════════════════════════════════════════
if page == "🏠  Home":
    st.markdown("""
    <div style='padding:48px 0 24px;'>
        <div style='font-size:11px; font-family:Space Mono; color:#3d5a80; letter-spacing:3px; margin-bottom:12px;'>ML-POWERED ASSESSMENT SYSTEM</div>
        <div style='font-size:52px; font-weight:800; color:#e8f0fe; line-height:1.1; letter-spacing:-2px;'>Operation Risk<br><span style='color:#63b3ff;'>Handling Index</span></div>
        <div style='font-size:16px; color:#4a6a9a; max-width:560px; margin-top:16px; line-height:1.7;'>
            Evaluate how ready professionals are for real-world job situations using unsupervised clustering and supervised classification models.
        </div>
    </div><hr>""", unsafe_allow_html=True)

    best_f1 = max(v['F1 Score'] for v in results.values())
    for col, (lbl, val, sub) in zip(st.columns(4), [
        ("REAL RESPONSES", str(original_count), "from survey"),
        ("TOTAL SAMPLES",  str(len(df)),         "after augmentation"),
        ("RISK CLUSTERS",  "03",                 "low / medium / high"),
        ("BEST F1 SCORE",  f"{best_f1:.2f}",     best_model[0]),
    ]):
        with col:
            st.markdown(f"<div class='metric-card'><div class='label'>{lbl}</div><div class='value'>{val}</div><div class='sub'>{sub}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr><div class='section-header'>How It Works</div><div class='section-sub'>PIPELINE OVERVIEW</div>", unsafe_allow_html=True)
    steps = [
        ("01","Data Collection","Real survey responses on 8 behavioral scenarios mapped to 1–5 scale","#63b3ff"),
        ("02","Augmentation","120 synthetic samples added via controlled noise injection","#a78bfa"),
        ("03","GMM Clustering","Gaussian Mixture Model assigns soft cluster probabilities → 3 risk groups","#34d399"),
        ("04","Classification","5 supervised models trained on GMM labels; best selected via F1","#fb923c"),
    ]
    for col, (num, title, desc, color) in zip(st.columns(4), steps):
        with col:
            st.markdown(f"<div class='metric-card' style='border-color:{color}22;'><div style='font-size:32px; font-weight:800; color:{color}; font-family:Space Mono;'>{num}</div><div style='font-size:15px; font-weight:700; color:#e8f0fe; margin:8px 0 6px;'>{title}</div><div style='font-size:13px; color:#4a6a9a; line-height:1.6;'>{desc}</div></div>", unsafe_allow_html=True)

    st.markdown("<hr><div class='section-header'>8 Behavioral Features</div>", unsafe_allow_html=True)
    feature_descs = [
        ("Unclear Instructions","How you respond when task requirements are ambiguous"),
        ("Deadline Pressure","Your approach when facing tight or shifting deadlines"),
        ("Ambiguity Comfort","Comfort level working with incomplete information"),
        ("Dependency Management","How you handle blockers from other team members"),
        ("Mistake Handling","Your response when you make errors at work"),
        ("Multi-tasking","Managing multiple concurrent responsibilities"),
        ("Unknown Tech","Adapting to unfamiliar tools or technologies"),
        ("Communication","Proactively sharing updates and blockers"),
    ]
    icons = ["🔵","🟣","🟢","🟠","🔴","🟡","🔷","🔶"]
    col_a, col_b = st.columns(2)
    for i, (feat, desc) in enumerate(feature_descs):
        with (col_a if i%2==0 else col_b):
            st.markdown(f"<div style='display:flex; align-items:flex-start; gap:14px; padding:14px 18px; background:#0d1525; border:1px solid #1a2a42; border-radius:12px; margin-bottom:10px;'><div style='font-size:20px; padding-top:2px;'>{icons[i]}</div><div><div style='font-size:14px; font-weight:700; color:#c9d6ea;'>{feat}</div><div style='font-size:12px; color:#3d5a80; margin-top:3px;'>{desc}</div></div></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# PAGE 2 — DATA & VISUALIZATIONS
# ═══════════════════════════════════════════════════════════════
elif page == "📊  Data & Visualizations":
    st.markdown("<div class='section-header'>Data & Visualizations</div><div class='section-sub'>EXPLORATORY ANALYSIS & CLUSTER INSIGHTS</div>", unsafe_allow_html=True)

    def dark_fig(w=10, h=7):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor('#0d1525'); ax.set_facecolor('#0d1525')
        return fig, ax

    tab1, tab2, tab3, tab4 = st.tabs(["📈 Correlation Heatmap","🫧 GMM Clusters (PCA)","📉 BIC Score","📦 Feature Distribution"])

    with tab1:
        fig, ax = dark_fig()
        corr = df.corr(numeric_only=True)
        sns.heatmap(corr, annot=True, cmap='YlOrRd', fmt='.2f', linewidths=0.5, ax=ax,
                    mask=np.triu(np.ones_like(corr, dtype=bool)),
                    annot_kws={'size':9,'color':'#e8f0fe'}, cbar_kws={'shrink':0.8})
        ax.set_title('Feature Correlation Matrix', color='#e8f0fe', fontsize=14, pad=16)
        ax.tick_params(colors='#6a8db8', labelsize=9); plt.xticks(rotation=35, ha='right')
        plt.tight_layout(); st.pyplot(fig)

    with tab2:
        X_s   = scaler.transform(df.values)
        probs  = gmm.predict_proba(X_s); conf = probs.max(axis=1)
        X_cl   = X_s[conf > 0.7]
        pca    = PCA(n_components=2); pca.fit(X_s)
        X_pca  = pca.transform(X_cl)
        lbl_cl = np.array([mapping[i] for i in gmm.predict(X_cl)])
        cen_c  = np.zeros_like(gmm.means_)
        for old, new in mapping.items(): cen_c[new] = gmm.means_[old]
        c_pca  = pca.transform(cen_c)

        fig, ax = dark_fig()
        for cl in [0,1,2]:
            idx = lbl_cl==cl
            ax.scatter(X_pca[idx,0], X_pca[idx,1], c=color_map[cl], alpha=0.65, s=40, label=labels_map[cl], edgecolors='none')
        for i in range(3):
            ax.scatter(c_pca[i,0], c_pca[i,1], c='white', s=220, marker='*', zorder=5)
            ax.annotate(["Low","Medium","High"][i], (c_pca[i,0],c_pca[i,1]), xytext=(10,8), textcoords='offset points', color='#e8f0fe', fontsize=10, fontweight='bold')
        ax.set_title('GMM Clustering — PCA Projection (confidence>0.7)', color='#e8f0fe', fontsize=13, pad=14)
        ax.set_xlabel('PC1', color='#6a8db8'); ax.set_ylabel('PC2', color='#6a8db8')
        ax.tick_params(colors='#4a6a9a'); ax.spines[:].set_color('#1e2d4a')
        ax.legend(facecolor='#0d1525', edgecolor='#1e2d4a', labelcolor='#c9d6ea')
        plt.tight_layout(); st.pyplot(fig)

    with tab3:
        X_s2   = scaler.transform(df.values)
        k_vals = list(range(1, 7))   # notebook: range(1, 7)
        bic    = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X_s2).bic(X_s2) for k in k_vals]
        fig, ax = dark_fig(8, 5)
        ax.plot(k_vals, bic, marker='o', color='#63b3ff', linewidth=2.5, markersize=8, markerfacecolor='#a78bfa')
        ax.axvline(x=3, color='#06d6a0', linestyle='--', alpha=0.6, label='Selected k=3')
        # annotate BIC values on each point
        for k, b in zip(k_vals, bic):
            ax.annotate(f'{b:.0f}', (k, b), textcoords='offset points', xytext=(0,10),
                        ha='center', color='#c9d6ea', fontsize=8)
        ax.set_xlabel('Number of Clusters', color='#6a8db8')
        ax.set_ylabel('BIC Score', color='#6a8db8')
        ax.set_title('BIC Score vs Number of GMM Components', color='#e8f0fe', fontsize=13, pad=14)
        ax.tick_params(colors='#4a6a9a', labelcolor='#6a8db8')
        ax.yaxis.set_tick_params(labelcolor='#6a8db8')
        ax.spines[:].set_color('#1e2d4a')
        ax.legend(facecolor='#0d1525', edgecolor='#1e2d4a', labelcolor='#c9d6ea')
        plt.tight_layout(); st.pyplot(fig)

    with tab4:
        fig, axes = plt.subplots(2, 4, figsize=(14,8))
        fig.patch.set_facecolor('#0d1525')
        fig.suptitle('Feature Distributions (Real + Augmented)', color='#e8f0fe', fontsize=14)
        palette = ['#63b3ff','#a78bfa','#34d399','#fb923c','#f472b6','#fbbf24','#60a5fa','#4ade80']
        for i, (cn, ax) in enumerate(zip(df.columns.tolist(), axes.flatten())):
            ax.set_facecolor('#0d1525')
            df[cn].value_counts().sort_index().plot(kind='bar', ax=ax, color=palette[i], edgecolor='none', alpha=0.85)
            ax.set_title(cn, color='#c9d6ea', fontsize=9, pad=6)
            ax.tick_params(colors='#4a6a9a', labelsize=8); ax.spines[:].set_color('#1e2d4a'); ax.set_xlabel('')
        plt.tight_layout(); st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════
# PAGE 3 — MODEL PERFORMANCE
# ═══════════════════════════════════════════════════════════════
elif page == "🤖  Model Performance":
    st.markdown("<div class='section-header'>Model Performance</div><div class='section-sub'>CROSS-VALIDATION RESULTS & ROC ANALYSIS</div>", unsafe_allow_html=True)

    bname = best_model[0]; bmet = results[bname]
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,#0f2a1f,#0a1e18); border:1px solid #1a5a3a; border-radius:16px; padding:24px 28px; margin-bottom:28px;'>
        <div style='font-size:11px; font-family:Space Mono; letter-spacing:2px; color:#2a7a4a;'>🏆 BEST MODEL</div>
        <div style='font-size:24px; font-weight:800; color:#06d6a0;'>{bname}</div>
        <div style='font-size:12px; color:#3a7a5a; font-family:Space Mono; margin-top:4px;'>
            F1: {bmet['F1 Score']:.4f} &nbsp;|&nbsp; Acc: {bmet['Accuracy']:.4f} &nbsp;|&nbsp; Prec: {bmet['Precision']:.4f} &nbsp;|&nbsp; Recall: {bmet['Recall']:.4f}
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("### 📋 5-Fold Cross-Validation Results")
    res_df = pd.DataFrame(results).T.reset_index().rename(columns={'index':'Model'}).sort_values('F1 Score', ascending=False)
    fig_t, ax_t = plt.subplots(figsize=(10,3)); fig_t.patch.set_facecolor('#0d1525'); ax_t.axis('off')
    cc = []
    for row in res_df.values.tolist():
        row_colors = []
        for j, v in enumerate(row):
            if j == 0:
                row_colors.append('#0d1525')  # Model name column
            else:
                fv = float(v)
                row_colors.append('#0f3d1f' if fv >= 0.8 else '#1a2f0f' if fv >= 0.6 else '#2a1515')
        cc.append(row_colors)
    tbl = ax_t.table(cellText=res_df.values.tolist(), colLabels=res_df.columns.tolist(), cellLoc='center', loc='center', cellColours=cc, colColours=['#0f2040']*len(res_df.columns))
    tbl.auto_set_font_size(False); tbl.set_fontsize(11); tbl.scale(1,2.0)
    for (r,c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#1e2d4a'); cell.set_text_props(color='#e8f0fe' if r==0 else '#c9d6ea')
    plt.tight_layout(); st.pyplot(fig_t)

    st.markdown("<hr>", unsafe_allow_html=True)
    col_roc, col_bar = st.columns(2)

    with col_roc:
        st.markdown("### 📉 ROC Curve")
        classes = np.unique(y_tr)
        y_bin   = label_binarize(y_te, classes=classes)
        y_score = best_model[1].predict_proba(X_te)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), y_score.ravel())
        roc_val = auc(fpr, tpr)
        fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor('#0d1525'); ax.set_facecolor('#0d1525')
        ax.plot(fpr, tpr, color='#63b3ff', linewidth=2.5, label=f'AUC = {roc_val:.2f}')
        ax.fill_between(fpr, tpr, alpha=0.07, color='#63b3ff')
        ax.plot([0,1],[0,1],'--', linewidth=1, color='#2a4060')
        ax.set_xlabel('FPR', color='#6a8db8'); ax.set_ylabel('TPR', color='#6a8db8')
        ax.set_title(f'ROC — {bname}', color='#e8f0fe', fontsize=12, pad=12)
        ax.tick_params(colors='#4a6a9a'); ax.spines[:].set_color('#1e2d4a')
        ax.legend(facecolor='#0d1525', edgecolor='#1e2d4a', labelcolor='#c9d6ea')
        plt.tight_layout(); st.pyplot(fig)

    with col_bar:
        st.markdown("### 📊 F1 Score Comparison")
        mnames = list(results.keys()); f1s = [results[m]['F1 Score'] for m in mnames]
        fig, ax = plt.subplots(figsize=(6,5)); fig.patch.set_facecolor('#0d1525'); ax.set_facecolor('#0d1525')
        bars = ax.barh(mnames, f1s, color=['#06d6a0' if m==bname else '#1e3a6e' for m in mnames], edgecolor='none', height=0.55)
        for bar, sc in zip(bars, f1s):
            ax.text(bar.get_width()+0.005, bar.get_y()+bar.get_height()/2, f'{sc:.3f}', va='center', color='#c9d6ea', fontsize=10)
        ax.set_xlabel('F1 Score (Weighted)', color='#6a8db8'); ax.set_title('Model Comparison', color='#e8f0fe', fontsize=12, pad=12)
        ax.tick_params(colors='#6a8db8'); ax.spines[:].set_color('#1e2d4a'); ax.set_xlim(0, max(f1s)+0.1)
        plt.tight_layout(); st.pyplot(fig)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### 🔀 K-Fold Visualization")
    # Exactly like notebook — full dataset scaled, full y labels
    X_s3   = scaler.transform(df.values)
    y_full = np.array([mapping[i] for i in np.argmax(gmm.predict_proba(X_s3), axis=1)])
    X_vis  = np.zeros((len(y_full), 1))
    skf_v  = StratifiedKFold(n_splits=5)
    fig, ax = plt.subplots(figsize=(12,3.5)); fig.patch.set_facecolor('#0d1525'); ax.set_facecolor('#0d1525')
    for i, (_, ti) in enumerate(skf_v.split(X_vis, y_full)):
        fa = np.zeros(len(y_full)); fa[ti] = 1
        ax.scatter(range(len(fa)), [i+1]*len(fa), c=['#63b3ff' if v==0 else '#ff6b6b' for v in fa], marker='s', s=12, alpha=0.7)
    ax.set_yticks(range(1,6)); ax.set_yticklabels([f"Fold {i}" for i in range(1,6)], color='#6a8db8')
    ax.set_xlabel("Sample Index", color='#6a8db8')
    ax.set_title("K-Fold Visualization  (🔴 Test  |  🔵 Train)", color='#e8f0fe', fontsize=12, pad=12)
    ax.tick_params(colors='#4a6a9a', labelcolor='#6a8db8'); ax.spines[:].set_color('#1e2d4a')
    plt.tight_layout(); st.pyplot(fig)

# ═══════════════════════════════════════════════════════════════
# PAGE 4 — RISK PREDICTOR
# ═══════════════════════════════════════════════════════════════
elif page == "🎯  Risk Predictor":
    st.markdown("<div class='section-header'>Risk Predictor</div><div class='section-sub'>ANSWER 8 BEHAVIORAL QUESTIONS TO GET YOUR RISK PROFILE</div>", unsafe_allow_html=True)
    st.markdown("<div style='background:#0d1525; border:1px solid #1a2a42; border-radius:12px; padding:16px 20px; margin-bottom:28px; font-size:13px; color:#4a6a9a; font-family:Space Mono;'>⚠️  Rate each scenario honestly on a scale of 1–5.<br>&nbsp;&nbsp;&nbsp;&nbsp;1 = Least effective &nbsp;|&nbsp; 5 = Most effective</div>", unsafe_allow_html=True)

    questions = [
        ("Unclear Instructions",  "1=Wait for someone else | 5=Clarify and suggest an approach"),
        ("Deadline Pressure",     "1=Panic and slow down | 5=Re-prioritize tasks calmly"),
        ("Ambiguity Comfort",     "1=Very uncomfortable | 5=Very comfortable"),
        ("Dependency Management", "1=Wait until others finish | 5=Actively coordinate and unblock"),
        ("Mistake Handling",      "1=Try to hide it | 5=Inform, fix, and prevent recurrence"),
        ("Multi-tasking",         "1=Miss some tasks | 5=Confidently re-prioritize"),
        ("Unknown Tech",          "1=Avoid it | 5=Learn quickly and solve"),
        ("Communication",         "1=Don't communicate | 5=Communicate early with solutions"),
    ]
    user_vals = []
    col_a, col_b = st.columns(2)
    for i, (qn, qh) in enumerate(questions):
        with (col_a if i%2==0 else col_b):
            val = st.slider(f"**{qn}**", 1, 5, 3, key=f"q_{i}", help=qh)
            st.markdown(f"<div style='font-size:11px; color:#2a4060; font-family:Space Mono; margin-top:-12px; margin-bottom:16px;'>{qh}</div>", unsafe_allow_html=True)
            user_vals.append(val)

    st.markdown("<hr>", unsafe_allow_html=True)
    _, btn_col, _ = st.columns([1,2,1])
    with btn_col:
        clicked = st.button("⚡  PREDICT MY RISK LEVEL", use_container_width=True)

    if clicked:
        inp       = scaler.transform(np.array([user_vals]))
        predicted = int(best_model[1].predict(inp)[0])   # numpy int → Python int
        proba     = best_model[1].predict_proba(inp)[0]

        rec_map = {
            0: ("You are not fully prepared for real-world job situations yet.",
                "Actively seek learning opportunities, engage in mock interviews, and participate in projects to gain practical experience. Embrace challenges as opportunities for growth."),
            1: ("You are moderately prepared but need improvement in some areas.",
                "Focus on areas where you lack confidence — structured problem-solving, asking clarifying questions. Consider mentorship or targeted skill development."),
            2: ("You are fully prepared for real-world job situations.",
                "Continue honing your skills, take on leadership roles, and mentor others. Explore advanced topics and continuously adapt to new challenges."),
        }
        base_msg, rec_msg = rec_map[predicted]

        st.markdown(f"""
        <div class='result-box' style='margin-top:28px;'>
            <div style='font-size:13px; font-family:Space Mono; letter-spacing:3px; color:#3d5a80;'>YOUR RISK PROFILE</div>
            <div class='risk-level' style='color:{color_map[predicted]};'>{emoji_map[predicted]}  {labels_map[predicted]}</div>
            <div style='margin:12px 0;'><span class='{badge_class[predicted]}'>{labels_map[predicted].upper()}</span></div>
            <div class='risk-desc'><b>{base_msg}</b><br><br>{rec_msg}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Model Confidence per Cluster:**", unsafe_allow_html=True)
        for ci, (cc, cname) in enumerate(zip(st.columns(3), labels_map.values())):
            pct = int(proba[ci]*100)
            with cc:
                st.markdown(f"""
                <div class='metric-card' style='text-align:center; border-color:{color_map[ci]}33;'>
                    <div style='font-size:11px; font-family:Space Mono; color:#4a6a9a; letter-spacing:1px;'>{cname.upper()}</div>
                    <div style='font-size:34px; font-weight:800; color:{color_map[ci]};'>{pct}%</div>
                    <div style='background:#1a2a42; border-radius:8px; height:6px; margin-top:8px;'>
                        <div style='width:{pct}%; height:100%; background:{color_map[ci]}; border-radius:8px;'></div>
                    </div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>**Your Response Profile vs Cluster Average:**", unsafe_allow_html=True)
        X_s4   = scaler.transform(df.values)
        y_f    = np.array([mapping[i] for i in np.argmax(gmm.predict_proba(X_s4), axis=1)])
        col_names = df.columns.tolist()
        df_tmp = df.copy(); df_tmp['cluster'] = y_f
        cl_avg = df_tmp.groupby('cluster')[col_names].mean().loc[predicted].values

        fig, ax = plt.subplots(figsize=(9,5)); fig.patch.set_facecolor('#0d1525'); ax.set_facecolor('#0d1525')
        xp = np.arange(len(col_names))
        ax.bar(xp-0.2, user_vals, 0.35, label='Your Scores',                  color='#63b3ff', alpha=0.85)
        ax.bar(xp+0.2, cl_avg,    0.35, label=f'{labels_map[predicted]} Avg', color=color_map[predicted], alpha=0.7)
        ax.set_xticks(xp); ax.set_xticklabels([c[:10] for c in col_names], rotation=25, ha='right', color='#6a8db8', fontsize=9)
        ax.set_ylabel('Score (1–5)', color='#6a8db8'); ax.set_title('Your Scores vs Cluster Average', color='#e8f0fe', fontsize=12, pad=12)
        ax.tick_params(colors='#4a6a9a'); ax.spines[:].set_color('#1e2d4a'); ax.set_ylim(0,5.8)
        ax.legend(facecolor='#0d1525', edgecolor='#1e2d4a', labelcolor='#c9d6ea')
        plt.tight_layout(); st.pyplot(fig)
