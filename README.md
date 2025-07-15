# FAST-LoRa: An Efficient Simulation Framework for Evaluating LoRaWAN Networks and Transmission Parameter Strategies

**FAST-LoRa** is a high-performance, lightweight simulation framework designed for the **fast and accurate evaluation of LoRaWAN networks** and **transmission parameter strategies**. It is tailored for scenarios with **stable traffic patterns** and **uplink-centric communication**, where full discrete-event simulation may be unnecessarily slow or complex.

Instead of relying on packet-level simulation, FAST-LoRa uses **analytical models** and **matrix-based operations** to reduce computational time while preserving strong approximation accuracy.

---

## 🚀 Key Features

- ⚡ **Fast and scalable**: Reduces simulation time by up to **1000×** compared to discrete-event simulators.
- 🎯 **Accurate approximation**:
  - **Packet Delivery Ratio (PDR)**: MAE = 0.0094
  - **Energy Efficiency (EE)**: MAE = 0.040 bits/mJ
- 🧠 **Efficient matrix-based gateway reception**
- 🧮 **No packet-level simulation** — analytical and lightweight
- ✅ **Validated** against a widely used LoRaWAN simulator across varied scenarios

---

## 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/fast-lora.git
cd fast-lora
pip install -r requirements.txt
