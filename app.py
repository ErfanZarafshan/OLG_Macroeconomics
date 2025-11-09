# app.py
# ECON7718 — OLG with Fiscal Policy: Q1–Q10 Interactive
# Streamlit app for figures, sliders, CSV/Excel downloads.

import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy import optimize

# ----------------------------
# Basic model primitives
# ----------------------------
def wage(k, alpha):
    return (1 - alpha) * (k ** alpha)

def interest(k, alpha, delta):
    return alpha * (k ** (alpha - 1.0)) - delta

def euler_ratio_R_equals_r(r, rho, theta, tau_k):
    # Γ = C2/C1 from Euler with R=r and tax on capital income
    # Γ = ((1-τ_k) r / (1+ρ))^(1/θ)
    if r <= 0:
        return np.nan
    Rtilde = (1 - tau_k) * r
    if Rtilde <= 0:
        return np.nan
    return (Rtilde / (1 + rho)) ** (1.0 / theta)

def saving_share_R_equals_r(r, rho, theta, tau_k):
    # s(r) from two-period CRRA with tax on capital income
    # s(r) = num / den where num = [(1-τ_k) r]^((1-θ)/θ)
    # and den = (1+ρ)^(1/θ) + num
    if r <= 0:
        return 0.0
    num = ((1 - tau_k) * r) ** ((1 - theta) / theta)
    den = (1 + rho) ** (1 / theta) + num
    return num / den

def lom_k_next(k, params):
    """
    Law of motion for k_{t+1} with fiscal policy:
    k_{t+1} = s(r_t) * [ (1-τ_w) w_t + κ(τ_w w_t + τ_k r_t k_t) ] / [ (1+g)(1+n) ]
    using r_t = α k^{α-1} - δ, w_t=(1-α)k^{α}
    """
    alpha, delta, rho, theta, tau_w, tau_k, kappa, g, n = params
    r = interest(k, alpha, delta)
    w = wage(k, alpha)
    s = saving_share_R_equals_r(r, rho, theta, tau_k)
    num = s * ((1 - tau_w) * w + kappa * (tau_w * w + tau_k * r * k))
    den = (1 + g) * (1 + n)
    return num / den

def steady_state_k(params, k0=1.0):
    f = lambda k: lom_k_next(k, params) - k
    return optimize.newton(f, k0)

# ----------------------------
# Streamlit page config
# ----------------------------
st.set_page_config(page_title="ECON7718 OLG — Q1–Q10", layout="wide")
st.title("ECON7718: OLG with Fiscal Policy — Interactive (Questions 1–10)")
st.title("Erfan Zarafshan, Ph.D. Student in Economcis at Louisiana State University")

with st.sidebar:
    st.header("Global Parameters")
    # Technology & demographics
    alpha = st.number_input(r"Capital share $\alpha$", 0.05, 0.95, 0.33, 0.01, format="%.2f")
    delta = st.number_input(r"Depreciation $\delta$", 0.0, 0.5, 0.05, 0.01, format="%.2f")
    g = st.number_input(r"Technology growth $g$", 0.0, 0.3, 0.02, 0.005, format="%.3f")
    n = st.number_input(r"Population growth $n$", 0.0, 0.3, 0.01, 0.005, format="%.3f")

    # Preferences
    rho = st.number_input(r"Discount rate $\rho$", 0.0, 0.5, 0.04, 0.005, format="%.3f")
    theta = st.number_input(r"CRRA $\theta$", 0.5, 10.0, 2.0, 0.1, format="%.1f")

    # Fiscal
    tau_k = st.number_input(r"Capital tax $\tau_k$", 0.0, 0.95, 0.10, 0.01, format="%.2f")
    tau_w = st.number_input(r"Labor tax $\tau_w$", 0.0, 0.95, 0.20, 0.01, format="%.2f")
    kappa = st.number_input(r"Young transfer share $\kappa$", 0.0, 1.0, 0.50, 0.05, format="%.2f")

    st.divider()
    st.caption("Use these parameters across all questions. You can change them any time.")

question = st.selectbox(
    "Jump to question",
    [f"Question {i}" for i in range(1, 11)]
)

# Pack parameters tuple for re-use
params = (alpha, delta, rho, theta, tau_w, tau_k, kappa, g, n)

# Utility: CSV & Excel download
def download_df(df, base_name):
    c1, c2 = st.columns(2)
    with c1:
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button(
            "Download CSV", data=csv_bytes,
            file_name=f"{base_name}.csv", mime="text/csv", key=f"{base_name}_csv"
        )
    with c2:
        xbuf = io.BytesIO()
        with pd.ExcelWriter(xbuf, engine="xlsxwriter") as w:
            df.to_excel(w, index=False, sheet_name="data")
        xbuf.seek(0)
        st.download_button(
            "Download Excel", data=xbuf.getvalue(),
            file_name=f"{base_name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key=f"{base_name}_xlsx"
        )

# ----------------------------
# Question 1
# ----------------------------
if question == "Question 1":
    st.subheader("Q1 — Euler, $\\Gamma=C_2/C_1$, Saving Share $s(r)$ (with $R=r$)")
    st.latex(r"\Gamma(r)=\left(\frac{(1-\tau_k)r}{1+\rho}\right)^{1/\theta},"
             r"\quad s(r)=\frac{\big[(1-\tau_k)r\big]^{\frac{1-\theta}{\theta}}}"
             r"{(1+\rho)^{\frac{1}{\theta}}+\big[(1-\tau_k)r\big]^{\frac{1-\theta}{\theta}}}")
    r = st.slider("Choose gross interest rate r", 0.01, 2.0, 0.60, 0.01)
    Gamma = euler_ratio_R_equals_r(r, rho, theta, tau_k)
    s_r = saving_share_R_equals_r(r, rho, theta, tau_k)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Γ = C₂/C₁", f"{Gamma:.6f}")
        st.metric("Saving share s(r)", f"{s_r:.6f}")
    with col2:
        # Plot Γ(r) and s(r)
        r_grid = np.linspace(0.02, 2.0, 300)
        Gamma_grid = ( ((1 - tau_k) * r_grid) / (1 + rho) ) ** (1/theta)
        s_grid = [saving_share_R_equals_r(x, rho, theta, tau_k) for x in r_grid]
        fig = plt.figure(figsize=(6,4))
        plt.plot(r_grid, Gamma_grid, label=r"$\Gamma(r)$")
        plt.plot(r_grid, s_grid, label=r"$s(r)$")
        plt.axvline(r, ls="--", lw=1)
        plt.xlabel("r"); plt.legend(); plt.tight_layout()
        st.pyplot(fig)

    df = pd.DataFrame({"r":[r],"Gamma":[Gamma],"s(r)":[s_r],
                       "rho":[rho],"theta":[theta],"tau_k":[tau_k]})
    download_df(df, "Q1_results")

# ----------------------------
# Question 2
# ----------------------------
elif question == "Question 2":
    st.subheader("Q2 — Budget constraint (PV) & $(C_1,C_2,S)$")
    st.latex(r"C_2=\Gamma C_1,\ \ S=(1-\tau_w)W_t+\eta_{y,t}-C_1,\ \ "
             r"C_2 = (1-\tau_k) r_{t+1} S + \eta_{o,t+1}")
    st.caption("Enter resource levels per effective worker (or levels with L=1).")
    W = st.number_input(r"$W_t$", 0.0, 10.0, 1.00, 0.1)
    eta_y = st.number_input(r"$\eta_{y,t}$", 0.0, 10.0, 0.00, 0.1)
    eta_o = st.number_input(r"$\eta_{o,t+1}$", 0.0, 10.0, 0.00, 0.1)
    r_next = st.slider(r"$r_{t+1}$", 0.01, 2.0, 0.60, 0.01)

    Gamma = euler_ratio_R_equals_r(r_next, rho, theta, tau_k)
    Rtilde = (1 - tau_k) * r_next
    # Lifetime PV of resources per young (with L=1):
    lifetime_R = (1 - tau_w) * W + eta_y + eta_o / Rtilde
    C1 = lifetime_R / (1 + Gamma / Rtilde)
    C2 = Gamma * C1
    S = (1 - tau_w) * W + eta_y - C1

    c1, c2, c3 = st.columns(3)
    c1.metric("C₁", f"{C1:.4f}")
    c2.metric("C₂", f"{C2:.4f}")
    c3.metric("Savings S", f"{S:.4f}")
    df = pd.DataFrame({
        "W_t":[W], "eta_y_t":[eta_y], "eta_o_t1":[eta_o], "r_next":[r_next],
        "Gamma":[Gamma], "C1":[C1], "C2":[C2], "S":[S],
        "rho":[rho], "theta":[theta], "tau_k":[tau_k], "tau_w":[tau_w]
    })
    download_df(df, "Q2_results")

# ----------------------------
# Question 3
# ----------------------------
elif question == "Question 3":
    st.subheader("Q3 — Factor prices from Cobb–Douglas")
    st.latex(r"w_t=(1-\alpha)k_t^{\alpha},\quad r_t=\alpha k_t^{\alpha-1}-\delta")
    k_t = st.slider(r"$k_t$", 0.01, 5.0, 0.50, 0.01)
    w_t = wage(k_t, alpha); r_t = interest(k_t, alpha, delta)
    st.metric("w_t", f"{w_t:.4f}")
    st.metric("r_t", f"{r_t:.4f}")
    df = pd.DataFrame({"k_t":[k_t],"w_t":[w_t],"r_t":[r_t],
                       "alpha":[alpha],"delta":[delta]})
    download_df(df, "Q3_results")

# ----------------------------
# Question 4
# ----------------------------
elif question == "Question 4":
    st.subheader("Q4 — Combine (Q1–Q3): $\\Gamma(r), s(r), (C_1,C_2,S)$ from factor prices")
    k_t = st.slider(r"$k_t$", 0.01, 5.0, 0.80, 0.01)
    w_t = wage(k_t, alpha); r_t = interest(k_t, alpha, delta)
    Gamma = euler_ratio_R_equals_r(r_t, rho, theta, tau_k)
    s_r = saving_share_R_equals_r(r_t, rho, theta, tau_k)
    st.latex(r"\Gamma(r_t),\ s(r_t)\ \text{and}\ (C_1,C_2,S)\ \text{as in Q2 with } r_{t+1}=r_t.")

    # Optionally compute (C1,C2,S) with given transfers W, eta's:
    with st.expander("Optional: compute (C₁,C₂,S) with resource inputs"):
        W = st.number_input(r"$W_t$", 0.0, 10.0, 1.00, 0.1, key="W_q4")
        eta_y = st.number_input(r"$\eta_{y,t}$", 0.0, 10.0, 0.00, 0.1, key="eta_y_q4")
        eta_o = st.number_input(r"$\eta_{o,t+1}$", 0.0, 10.0, 0.00, 0.1, key="eta_o_q4")
        if (1 - tau_k) * r_t > 0:
            Rtilde = (1 - tau_k) * r_t
            lifetime_R = (1 - tau_w) * W + eta_y + eta_o / Rtilde
            C1 = lifetime_R / (1 + Gamma / Rtilde)
            C2 = Gamma * C1
            S = (1 - tau_w) * W + eta_y - C1
            st.write(f"C₁={C1:.4f}, C₂={C2:.4f}, S={S:.4f}")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("w_t", f"{w_t:.4f}")
        st.metric("r_t", f"{r_t:.4f}")
    with col2:
        st.metric("Γ(r_t)", f"{Gamma:.6f}")
        st.metric("s(r_t)", f"{s_r:.6f}")

    df = pd.DataFrame({"k_t":[k_t],"w_t":[w_t],"r_t":[r_t],
                       "Gamma":[Gamma],"s(r)":[s_r]})
    download_df(df, "Q4_results")

# ----------------------------
# Question 5
# ----------------------------
elif question == "Question 5":
    st.subheader("Q5 — Visualize $\\Gamma(r)$ and $s(r)$ vs r")
    r_grid = np.linspace(0.02, 2.0, 300)
    Gamma_grid = ( ((1 - tau_k) * r_grid) / (1 + rho) ) ** (1/theta)
    s_grid = [saving_share_R_equals_r(x, rho, theta, tau_k) for x in r_grid]
    fig = plt.figure(figsize=(7,4))
    plt.plot(r_grid, Gamma_grid, label=r"$\Gamma(r)$", lw=2)
    plt.plot(r_grid, s_grid, label=r"$s(r)$", lw=2)
    plt.axhline(1, color='gray', ls='--', lw=1)
    plt.xlabel("r"); plt.legend(); plt.tight_layout()
    st.pyplot(fig)
    df = pd.DataFrame({"r":r_grid, "Gamma":Gamma_grid, "s(r)":s_grid})
    download_df(df, "Q5_curves")

# ----------------------------
# Question 6
# ----------------------------
elif question == "Question 6":
    st.subheader("Q6 — Law of Motion with Fiscal Policy")
    st.latex(r"k_{t+1}=\frac{s(r_t)\big[(1-\tau_w)w_t+\kappa(\tau_w w_t+\tau_k r_t k_t)\big]}{(1+g)(1+n)}")
    k_t = st.slider(r"$k_t$", 0.01, 5.0, 0.80, 0.01, key="k_q6")
    k_next_val = lom_k_next(k_t, params)
    st.metric(r"$k_{t+1}$", f"{k_next_val:.4f}")

    # Plot k_{t+1}(k_t) with 45-degree line
    k_grid = np.linspace(0.02, 5, 300)
    kn_grid = np.array([lom_k_next(kk, params) for kk in k_grid])
    fig = plt.figure(figsize=(7,4))
    plt.plot(k_grid, kn_grid, lw=2, label=r"$k_{t+1}(k_t)$")
    plt.plot(k_grid, k_grid, "--", color="gray", label=r"45° line")
    plt.xlabel(r"$k_t$"); plt.ylabel(r"$k_{t+1}$")
    plt.legend(); plt.tight_layout()
    st.pyplot(fig)
    df = pd.DataFrame({"k_t":k_grid,"k_{t+1}":kn_grid})
    download_df(df, "Q6_kmap")

# ----------------------------
# Question 7
# ----------------------------
elif question == "Question 7":
    st.subheader("Q7 — Plot $k_{t+1}(k_t)$ and 45° line, mark $k^*$")
    k_grid = np.linspace(0.02, 5, 400)
    kn_grid = np.array([lom_k_next(kk, params) for kk in k_grid])
    # steady state
    try:
        k_star = steady_state_k(params, k0=1.0)
    except Exception:
        k_star = np.nan
    fig = plt.figure(figsize=(7,4))
    plt.plot(k_grid, kn_grid, lw=2, label=r"$k_{t+1}(k_t)$")
    plt.plot(k_grid, k_grid, "--", color="gray", label=r"45° line")
    if np.isfinite(k_star):
        plt.scatter([k_star],[k_star], s=60, color="red", zorder=5, label=rf"$k^* \approx {k_star:.3f}$")
    plt.xlabel(r"$k_t$"); plt.ylabel(r"$k_{t+1}$")
    plt.legend(); plt.tight_layout()
    st.pyplot(fig)
    
    
    df = pd.DataFrame({
        "k_t": k_grid,
        "k_{t+1}": kn_grid
    })
    df["k_star"] = k_star  # assign as a constant column




# ----------------------------
# Question 8
# ----------------------------
elif question == "Question 8":
    st.subheader("Q8 — Compute steady state $k^*$")
    st.caption("Solve k* from k_{t+1}(k)=k using a root finder.")
    try:
        k_star = steady_state_k(params, k0=1.0)
        st.metric("Steady state k*", f"{k_star:.6f}")
    except Exception as e:
        st.error(f"Root finding failed: {e}")
        k_star = np.nan

    # Visual check
    k_grid = np.linspace(0.05, 5, 300)
    kn_grid = np.array([lom_k_next(kk, params) for kk in k_grid])
    fig = plt.figure(figsize=(7,4))
    plt.plot(k_grid, kn_grid, lw=2, label=r"$k_{t+1}(k_t)$")
    plt.plot(k_grid, k_grid, "--", color="gray", label=r"45° line")
    if np.isfinite(k_star):
        plt.axvline(k_star, color="red", ls=":", lw=1.5, label=rf"$k^* \approx {k_star:.3f}$")
    plt.legend(); plt.tight_layout()
    st.pyplot(fig)
    df = pd.DataFrame({"k_t": k_grid, "k_{t+1}": kn_grid, "k_star": [k_star]})
    download_df(df, "Q8_kstar")



# ----------------------------
# Question 9
# ----------------------------
elif question == "Question 9":
    st.subheader("Q9 — Transitional dynamics from $k_0=0.5 k^*$")
    T = st.slider("Horizon (T)", 5, 200, 50, 1)
    try:
        k_star = steady_state_k(params, k0=1.0)
    except Exception:
        st.error("Could not find steady state."); k_star = np.nan

    if np.isfinite(k_star):
        k_path = np.zeros(T)
        k_path[0] = 0.5 * k_star
        for t in range(T-1):
            k_path[t+1] = lom_k_next(k_path[t], params)
        fig = plt.figure(figsize=(7,4))
        plt.plot(range(T), k_path, lw=2, label=r"$k_t$")
        plt.axhline(k_star, color="red", ls=":", label=rf"$k^*={k_star:.3f}$")
        plt.xlabel("t"); plt.ylabel(r"$k_t$"); plt.legend(); plt.tight_layout()
        st.pyplot(fig)
        df = pd.DataFrame({"t":np.arange(T), "k_t":k_path, "k_star":[k_star]*T})
        download_df(df, "Q9_path")
    else:
        st.info("Provide parameters that deliver a finite steady state.")

# ----------------------------
# Question 10
# ----------------------------
elif question == "Question 10":
    st.subheader("Q10 — Policy experiments (No Policy vs A vs B) & growth (low/high)")
    st.caption("We compute k* for each policy under g∈{low, high} and plot steady states & transition paths.")
    # Define policies
    policies = [
        {"label": "No Policy", "tau_k": 0.0, "tau_w": 0.0, "kappa": 0.0},
        {"label": "Policy A",  "tau_k": 0.2, "tau_w": 0.1, "kappa": 0.5},
        {"label": "Policy B",  "tau_k": 0.3, "tau_w": 0.2, "kappa": 0.8},
    ]
    g_low = st.number_input("Low growth g_low", 0.0, 0.3, 0.02, 0.005, format="%.3f")
    g_high = st.number_input("High growth g_high", 0.0, 0.3, 0.05, 0.005, format="%.3f")

    def k_ss_given(tk, tw, kap, gg):
        p = (alpha, delta, rho, theta, tw, tk, kap, gg, n)
        return steady_state_k(p, k0=1.0)

    # 1) Bar chart of steady states
    results = {"low":{}, "high":{}}
    for p in policies:
        results["low"][p["label"]]  = k_ss_given(p["tau_k"], p["tau_w"], p["kappa"], g_low)
        results["high"][p["label"]] = k_ss_given(p["tau_k"], p["tau_w"], p["kappa"], g_high)

    labels = [p["label"] for p in policies]
    low_vals = [results["low"][lab] for lab in labels]
    high_vals = [results["high"][lab] for lab in labels]

    fig = plt.figure(figsize=(7,4))
    x = np.arange(len(labels)); width = 0.35
    plt.bar(x - width/2, low_vals, width, label=fr"Low growth (g={g_low:.2f})")
    plt.bar(x + width/2, high_vals, width, label=fr"High growth (g={g_high:.2f})")
    plt.xticks(x, labels); plt.ylabel(r"$k^*$"); plt.title("Steady-state comparison")
    plt.legend(); plt.tight_layout()
    st.pyplot(fig)

    # 2) Combined convergence figures (low and high), like your compact style
    T = st.slider("Transition horizon (T)", 5, 100, 20, 1)
    colors = {"No Policy":"#e69f00","Policy A":"#009e73","Policy B":"#56b4e9"}
    markers = {"No Policy":"o","Policy A":"s","Policy B":"D"}

    def k_next_policy(k, tk, tw, kap, gg):
        return lom_k_next(k, (alpha, delta, rho, theta, tw, tk, kap, gg, n))

    # low growth
    fig = plt.figure(figsize=(7,4))
    for p in policies:
        k_star = results["low"][p["label"]]
        k_path = np.zeros(T+1); k_path[0] = 0.5 * k_star
        for t in range(T):
            k_path[t+1] = k_next_policy(k_path[t], p["tau_k"], p["tau_w"], p["kappa"], g_low)
        plt.plot(range(T+1), k_path, label=p["label"], color=colors[p["label"]],
                 marker=markers[p["label"]], lw=2)
        plt.hlines(k_star, 0, T, colors=colors[p["label"]], linestyles="--", lw=1.3)
    plt.title(fr"Transitions — Low growth (g={g_low:.2f})"); plt.xlabel("t"); plt.ylabel(r"$k_t$")
    plt.legend(); plt.grid(True, ls=":"); plt.tight_layout()
    st.pyplot(fig)

    # high growth
    fig = plt.figure(figsize=(7,4))
    for p in policies:
        k_star = results["high"][p["label"]]
        k_path = np.zeros(T+1); k_path[0] = 0.5 * k_star
        for t in range(T):
            k_path[t+1] = k_next_policy(k_path[t], p["tau_k"], p["tau_w"], p["kappa"], g_high)
        plt.plot(range(T+1), k_path, label=p["label"], color=colors[p["label"]],
                 marker=markers[p["label"]], lw=2)
        plt.hlines(k_star, 0, T, colors=colors[p["label"]], linestyles="--", lw=1.3)
    plt.title(fr"Transitions — High growth (g={g_high:.2f})"); plt.xlabel("t"); plt.ylabel(r"$k_t$")
    plt.legend(); plt.grid(True, ls=":"); plt.tight_layout()
    st.pyplot(fig)

    # Data tables for download
    df_low = pd.DataFrame({"Policy":labels, "k*_low":low_vals})
    df_high = pd.DataFrame({"Policy":labels, "k*_high":high_vals})
    download_df(df_low.merge(df_high, on="Policy"), "Q10_kstars")
