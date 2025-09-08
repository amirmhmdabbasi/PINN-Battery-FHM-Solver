# ⚡ PINN-Battery-FHM-Solver

This project presents a solver based on the **Physics-Informed Neural Networks (PINNs)** framework to solve the **Full Homogenized Macroscale (FHM)** model of lithium-ion batteries. The model takes basic battery inputs, such as **time**, **C-rate**, and **temperature**, and computes key electrochemical outputs, including **concentration profiles** and **potentials**.

---

## 🔋 Data Generation

The dataset is generated using simulations of the FHM model, originally introduced by Arunachalam in  
_"A New Multiscale Modeling Framework for Lithium-Ion Battery Dynamics: Theory, Experiments, and Comparative Study with the Doyle-Fuller-Newman Model"_.

The model is implemented in **COMSOL Multiphysics** and run under various C-rates. Simulation outputs include:

-  Solid-phase and electrolyte-phase concentrations  
-  Electrochemical potentials  
-  Intercalation current density

---

## 📊 Outputs & Evaluation

These outputs are used to generate:

- 🔋 Battery discharge curves  
- 📉 State of Charge (SOC) variation graphs during discharge cycles

The PINN framework is built using **PyTorch**, enabling a physics-informed approach to model training and prediction with reasonable accuracy.

---

## 📄 Full Framework Description

A complete description of the proposed framework—including battery parameters, physical model, and setup procedure for accurate estimation of discharge curves and SOC—is available in our paper:

_Real-Time Discharge Curve and State of Charge Estimation of Lithium-Ion Batteries via a Physics-Informed Full Homogenized Macroscale Model_ - Journal of Energy Storage, AmirMohammad Abbasi, Ayat Gharehghani, Amin Andwari
DOI: https://doi.org/10.1016/j.est.2025.118307

## 📂 Data Access

The main dataset used in this work can be accessed via the following link:  
🔗 [Download from Google Drive](https://drive.google.com/file/d/1UT6MDHu-fLcj3zq5KsDmT8Ih0KgCv-4w/view?usp=sharing)

If the link is unavailable or you encounter any issues, feel free to reach out via email:  
📧 amir.m.abbasi78@gmail.com

## 🧠 Code Structure & Execution Guide

This repository contains several modular `.py` files, each responsible for a specific part of the PINN-based battery modeling workflow. To run the full pipeline and reproduce results step-by-step, we recommend combining these files into separate cells within a **JupyterLab** or **Google Colab** environment.

Below is the recommended execution order and a brief description of each file:


### 1️⃣ `data_loader.py`
📦 **Purpose**:  
- Imports required libraries  
- Defines battery physical parameters  
- Organizes, preprocesses, splits, and normalizes the dataset for model training

---

### 2️⃣ `battery_model.py`
⚙️ **Purpose**:  
- Implements residual functions based on the **FHM** battery model  
- Includes governing equations, boundary and initial conditions  
- Provides external functions to compute dependent battery parameters

---

### 3️⃣ `pinn_framework.py`
🏗️ **Purpose**:  
- Defines the main architecture of the proposed **PINN** framework  
- Built using PyTorch for flexible and physics-aware modeling

---

### 4️⃣ `loss_functions.py`
📉 **Purpose**:  
- Calculates loss functions using both physics-based residuals and data-driven components  
- Integrates outputs from `battery_model.py` and training data

---

### 5️⃣ `train.py`
🎯 **Purpose**:  
- Contains the pretraining routine and main training loop  
- Optimizes the PINN model using defined losses and training data

---

### 6️⃣ `evaluate.py`
📊 **Purpose**:  
- Computes evaluation metrics such as **RMSE**, **MAE**, and **R²**  
- Generates discharge curves and **State of Charge (SOC)** variation graphs for model validation

---

You can follow this order to execute the code and reproduce the results seamlessly. Each file is self-contained and documented for clarity. If you'd like a notebook version or a minimal `main.py` to orchestrate the workflow, feel free to reach out or adapt the structure above.

