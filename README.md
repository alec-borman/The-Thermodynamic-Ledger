# Thermodynamic Ledger Core (TLC)

**Status:** Pre-Alpha / Experimental  
**License:** MIT  

## The Mission
To establish a "Proof-of-Physics" consensus mechanism for Decentralized Science (DeSci). 
This repository implements the **Bit-Perfect Verification Kernel (BPVK)**—a CUDA-based engine designed to execute integer-based physics simulations with 100% deterministic reproducibility across different GPU architectures (NVIDIA A100, H100, RTX Consumer Cards).

## The Problem: Determinism
Standard floating-point arithmetic (IEEE 754) is non-associative `(a+b)+c != a+(b+c)` on parallel GPUs due to thread scheduling differences. This makes it impossible to verify scientific computations on a blockchain without bit-level drift.

## The Solution: Integer Physics
We reject `float`. We define a Planck Unit for the simulation (1 femtometer) and execute all molecular dynamics using 64-bit fixed-point integers and a **Canonical Summation Tree (CST)** kernel that enforces rigid reduction topology.

## Building & Testing

### Prerequisites
- NVIDIA GPU (Compute Capability 6.0+)
- CUDA Toolkit 11+
- CMake 3.18+
- C++17 Compiler

### Build
```bash
mkdir build && cd build
cmake ..
make
