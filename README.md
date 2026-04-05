# CTES-framework

**Control-Theoretic Equilibrium Selection (CTES)** — a novel framework that repurposes LQR optimal control theory as a global optimization proxy. Given a multimodal nonlinear function with multiple local minima, CTES uses the solution to the Algebraic Riccati Equation (ARE) at each equilibrium to rank candidate solutions and identify the global minimum.

> This work connects LQR energy (−trace(P)) to basin depth in nonlinear landscapes, achieving perfect Spearman rank correlation on both negative-valued (Gaussian wells) and non-negative (Rastrigin) function classes via a unified depth-normalized encoding.

---

## Core Idea

At each local minimum x\*, linearize the optimization landscape and pose an LQR problem:

```
A = −H(x*)        (gradient flow dynamics)
B = I             (full actuation)
Q = depth(x*) · H (depth-encoded state cost)
R = I             (unit control cost)
```

Solve the continuous Algebraic Riccati Equation for P, then use **−trace(P)** as an energy proxy. The global minimum receives the lowest energy.

**Unified depth encoding (v8):**
```
depth(x*) = (f_max − f(x*)) / (f_max − f_min + ε)  ∈ [0, 1]
```
This is sign-agnostic — works for f < 0 (Gaussian wells) and f ≥ 0 (Rastrigin) without modification.

---

## Repository Structure

```
ctes-framework/
│
├── lqr_optim_v4_gaussian.m      # 1D: Flipped proxy (−P), 5-well Gaussian, rho = +1
├── lqr_optim_v5_2d.m            # 2D extension: proxy = −trace(P), 5-well landscape
├── lqr_optim_v7_rastrigin.m     # Rastrigin stress test: inverted Q encoding (Fix 3)
├── lqr_optim_v8_unified.m       # Unified formula: both Gaussian + Rastrigin, rho > 0.99
└── ctes_3link_robot.m           # Application: 3-link robot IK equilibrium selection
```

---

## Files

### `lqr_optim_v4_gaussian.m`
1D optimization on a 5-well Gaussian landscape where the global minimum is **not** the stiffest well. Introduces the flipped proxy −P (fixing the sign inversion from v3). Achieves ρ = 1.0.

### `lqr_optim_v5_2d.m`
Extends the framework to 2D scalar functions f(x₁, x₂). Hessian is now a 2×2 matrix; proxy is −trace(P). Demonstrates the approach scales to vector state spaces.

### `lqr_optim_v7_rastrigin.m`
Stress test on the Rastrigin function (f ≥ 0 everywhere, 10+ local minima on [−5, 5]). Uses an inverted Q encoding: Q = 1/(|f(x\*)| + ε) · H so that the global minimum (f ≈ 0) receives the largest Q.

### `lqr_optim_v8_unified.m`
Unifies v4 and v7 into a single sign-agnostic formula using normalized range depth. Runs both test classes back-to-back and confirms ρ > 0.99 on each. **This is the recommended starting point.**

### `ctes_3link_robot.m`
Applies the CTES framework to inverse kinematics equilibrium selection for a 3-link planar robot arm. Multiple IK solutions exist for a given end-effector target; CTES selects the energetically optimal configuration. Includes PD controller simulation and animation.

---

## Requirements

- MATLAB R2019b or later
- Control System Toolbox (for `care`)
- No additional toolboxes required for v4/v5/v7/v8

---

## Results Summary

| Script | Function Class | Spearman ρ | Global Correct |
|--------|---------------|------------|----------------|
| v4 | 5-well Gaussian (1D) | 1.0000 | ✓ |
| v5 | 5-well Gaussian (2D) | 1.0000 | ✓ |
| v7 | Rastrigin (Fix 3) | >0.95 | ✓ |
| v8 | Gaussian + Rastrigin (unified) | >0.99 | ✓ |
| CTES Robot | 3-link IK selection | — | ✓ |

---

## Background & Motivation

Standard global optimization methods (basin-hopping, simulated annealing, genetic algorithms) use stochastic search without exploiting local geometric structure. CTES takes a different approach: instead of escaping local minima randomly, it **ranks** them using a control-theoretic energy measure derived from the local curvature and depth of each basin.

The key insight is that the LQR cost-to-go P encodes both curvature (via H in Q) and depth (via the depth scalar), so −trace(P) serves as a proxy for "how globally important is this equilibrium?"

This connects to ideas in:
- Lyapunov stability theory (P as a Lyapunov matrix)
- LQR as an energy minimization problem
- Basin geometry in dynamical systems

---

## Author

**Saurav Avachat**  
B.Tech Electronics & Communication Engineering, VNIT Nagpur  
Incoming MSc Systems & Control, TU Delft (DCSC), 2026
