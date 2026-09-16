# Advanced Machine Learning (ST7085CEM) Coursework

**MSc Advanced Machine Learning — Final Coursework Project**  
**Authors:** [Sufi Dhungel](https://github.com/Sufi-123), [Jyoti Bista](https://github.com/jyotibista), [Rakshya Nakarmi](https://github.com/rakshyanki4151-web)  
**Video Demonstration:** 📺 [AML Coursework Video Demonstration (YouTube)](https://www.youtube.com/watch?v=ZyGZq_3H2c0)  

This repository contains the complete, reproducible source code, datasets, evaluation pipelines, and interactive applications for both **Task 1** and **Task 2**.

---

## 📂 Repository Structure

The project is cleanly structured into two dedicated directories:

```text
AML-Coursework-Task1-Task2/
│
├── 📁 Task_1_GPR_Renewable_Forecasting/    # Task 1: Python / Machine Learning Pipeline
│   ├── app/                               # Streamlit Interactive Dashboard
│   ├── src/                               # Core Python ML modules (GPR, PSO, GPC, Evaluators)
│   ├── data/                              # Hourly meteorological datasets (Jumla, Karnali)
│   ├── outputs/                           # Trained models (.joblib), metrics JSON/CSV, figures
│   ├── figures/                           # System architecture and workflow diagrams
│   ├── run.py                             # Master pipeline driver (python run.py)
│   ├── requirements.txt                   # Pinned dependencies
│   └── README.md                          # Comprehensive Task 1 Technical Documentation
│
├── 📁 Task_2_Fuzzy_Logic_Controller/       # Task 2: MATLAB / Evolutionary Fuzzy Control
│   ├── part1_flc/                         # Mamdani FLC design & scenario analysis
│   ├── part2_ga_tuning/                   # Genetic Algorithm chromosome tuning
│   ├── part3_optimizer_comparison/        # CEC'2005 GA vs. PSO benchmark study
│   ├── shared/                            # Custom GA, PSO, and thermal room simulation
│   ├── run_all_task2.m                    # Master one-command execution driver
│   ├── README.md                          # Technical overview of FLC & GA architecture
│   └── HOW_TO_RUN.md                      # Step-by-step MATLAB execution instructions
│
└── 📄 README.md                           # Master Project Overview (this file)
```

---

## ⚡ Task 1: Physics-Informed GPR with PSO for Renewable Energy Forecasting

- **Domain:** Renewable microgrid energy management in **Jumla, Karnali Province, Nepal (29.267°N, 82.183°E, 2,300m elevation)**.
- **Framework:** Physics-Informed Gaussian Process Regression (GPR) with hand-coded Particle Swarm Optimization (PSO).
- **Core Features:**
  - 52-column design matrix combining cyclical solar geometry, physical clear-sky and thermal priors, autoregressive lags, and rolling averages.
  - Rigorous target-by-target benchmarking: Persistence, Ridge Regression, SARIMAX(2,0,2), GPR (RBF, Matérn, Periodic kernels), and PSO-tuned GPR.
  - Discrete dispatch classification (Low/Medium/High generation bands) via Gaussian Process Classification (GPC).
  - Walk-forward expanding window validation (Design 1 & Design 2) with Diebold–Mariano statistical significance testing.
  - Edge-hardware feasibility verification confirming sub-millisecond execution ($0.135\text{ ms}$) and $<0.2\text{ MB}$ memory footprint on Raspberry Pi 4.
  - Real-time interactive decision-support interface built with Streamlit.

👉 **See detailed instructions and methodology:** [`Task_1_GPR_Renewable_Forecasting/README.md`](Task_1_GPR_Renewable_Forecasting/README.md)

---

## 🎛️ Task 2: Fuzzy Logic Optimized Controller for an Intelligent Assistive Care Environment

- **Domain:** Environmental climate control (fuzzy PD thermostat) for an assisted living residence.
- **Framework:** 2-input, 1-output Mamdani Fuzzy Inference System (FLC) with real-coded Genetic Algorithm (GA) optimization.
- **Core Features:**
  - 15-rule symmetric rule base with centroid defuzzification controlling a combined heating/cooling actuator.
  - Real-coded Genetic Algorithm using a 6-gene affine chromosome representation, achieving a **42.1% reduction** in tracking MSE while guaranteeing full universe-of-discourse coverage.
  - Independent Part 3 optimizer benchmark comparing custom GA vs. custom PSO on CEC'2005 Shifted Sphere (F1) and Shifted Rastrigin (F9) functions at dimensions $D=2$ and $D=10$ over 15 repetitions.

👉 **See detailed instructions and methodology:** [`Task_2_Fuzzy_Logic_Controller/README.md`](Task_2_Fuzzy_Logic_Controller/README.md)  
👉 **How to run in MATLAB:** [`Task_2_Fuzzy_Logic_Controller/HOW_TO_RUN.md`](Task_2_Fuzzy_Logic_Controller/HOW_TO_RUN.md)

---

## 🚀 Quick Execution Guide

### Task 1 (Python 3.9+)
```bash
cd Task_1_GPR_Renewable_Forecasting
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
pip install -r requirements.txt
python run.py                      # Runs the full training & evaluation pipeline
streamlit run app/streamlit_app.py # Launches the interactive web dashboard
```

### Task 2 (MATLAB)
```matlab
cd Task_2_Fuzzy_Logic_Controller
run_all_task2   % Executes Part 1, Part 2, and Part 3 end-to-end
```
