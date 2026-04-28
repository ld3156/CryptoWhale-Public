# 🐋 Do Whales Move Prices? — Can We Trade It?

**Columbia Business School – Master Thesis (2026)**  


> This project investigates whether large crypto investors ("whales") have predictive power over market prices — and whether their behavior can be systematically transformed into tradable alpha signals.

---

## 📌 Overview

Crypto markets are highly fragmented and often driven by large participants.  
This project aims to:

- Monitor **real-time whale activities**
- Construct **whale-based trading signals**
- Integrate them with **traditional CTA strategies**
- Evaluate whether they provide **incremental alpha**

> Key Result:  
> Yes — whale signals add statistically significant alpha beyond traditional CTA factors.

---

## 🧠 Key Idea

**whales** as large investors with typically large notional positionbook.

We extract whale activity data from live fetching Coinglass API and generate from several traditional signals:

- Cumulative Volume Delta (CVD)
- Large Transaction Volume (LTV)
- Position changes, leverage, and flow dynamics
- Majority and Minority Actions
- ...

These features are then aggregated into **minute-level signals** and used for trading.

---

## ⚙️ Data Pipeline

We build a **real-time data ingestion system**:

- Source: CoinGlass <-- Hyperliquid API
- Frequency: **1-minute level**
- Infrastructure: AWS (Lambda + S3)

### Workflow
API → AWS Lambda → Data Transformation → S3 Storage → Research / Backtesting


The dataset includes:

- Whale positions
- Transaction alerts
- Price & volume data
- 100+ engineered features

> Time series starts from **March 2026**, aligned with cleaned market data.

---

## 🧩 Whale Behavior Modeling

We classify whales into behavioral archetypes:

- **Fast Speculators**
  - High turnover, short holding time
- **Hedger-like**
  - Market-neutral positioning
- **High Conviction Traders**
  - Large exposure, long holding period

This allows us to convert **raw wallet behavior → interpretable trading signals**.

---

## 📊 Feature Engineering

### Whale Signal Categories

We construct signals across 7 dimensions:

- **Flow** – order direction & size  
- **Position Book** – net exposure, leverage  
- **Attention** – alert frequency  
- **Momentum** – price + whale flow dynamics  
- **Reverting** – mean reversion behavior  
- **Volatility** – dispersion of whale activity  
- **Actions** – open vs close behavior  

> We select **20+ high-performing signals** based on in-sample tests.

---

## 📈 Baseline Strategy

We build a **CTA-style strategy** using:

- Trend
- Momentum
- Mean Reversion
- Breakout
- Volatility
- Volume
- Adaptive signals

→ Select **Top 30 factors** from in-sample performance

---

## 🚀 Results

### 1. Whale Signals Alone

- Positive cumulative returns
- Robust performance across volatility regimes

### 2. CTA + Whale Signals

- Improved **Sharpe ratio**
- Better **drawdown profile**
- Stronger **out-of-sample performance**

### 3. Statistical Evidence

- Whale signals explain **residual returns**
- Significant coefficients after controlling CTA factors

> Whale signals provide **incremental, independent alpha**

---

## 🔬 Economic Interpretation

Examples:

- **Abnormal inflows → price pressure**
- **Extreme activity → reversal signals**
- **Leverage stress → market instability indicator**

---

## 🏗️ Research Pipeline

We built a fully reusable research framework:

- Intraday signal construction
- Factor formula language (WorldQuant-style)
- Time-series & cross-sectional operations
- Backtesting & visualization tools
- Machine learning integration

---

## ⚠️ Limitations

- “Market-neutral” assumption may be overstated
- Potential data-snooping / model selection bias
- Practical constraints (costs, execution) require further validation

---

## 🔮 Future Work

- Extend to multi-asset strategies
- Improve execution modeling
- Refine economic interpretation of signals
- Integrate alternative data (sentiment, derivatives)

---

## 📁 Disclaimer

- All data is from **public sources or Coinglass API access**
- Results are for **research purposes only**
- For those who may want to replicate, please reach out to Coinglass team and ask for APIKEY.
- For the dataset in this research, please reach out to Li (ldai26@gsb.columbia.edu) or Peter (gshen26@gsb.columbia.edu) for more info.
---

## 👥 Authors

Github Repo - Li Dai, Peter Shen (Columbia Business School, 2026)

---
