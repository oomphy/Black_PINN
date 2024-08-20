# Solving Black-Scholes Equation for American Option Pricing using Physics-Informed Neural Networks (PINNs)

## Overview

This project demonstrates the application of Physics-Informed Neural Networks (PINNs) to solve the Black-Scholes partial differential equation (PDE) for American option pricing. PINNs leverage neural networks to learn the solution to PDEs by incorporating the physical laws and boundary conditions directly into the training process. This approach provides a powerful and efficient method for pricing American options, which involve early exercise features not easily handled by traditional numerical methods.

## Black-Scholes Equation for American Options

The Black-Scholes equation is a fundamental PDE used in financial mathematics to model the price of European options. For American options, which allow for early exercise, the problem becomes more complex due to the option’s early exercise feature. The PINN approach solves this by learning a neural network that approximates the solution to the Black-Scholes PDE while adhering to the American option’s boundary conditions.

## Key Features

- **Physics-Informed Approach**: Utilizes neural networks to solve the Black-Scholes PDE by incorporating the physical laws and boundary conditions directly into the loss function.
- **Early Exercise Feature**: Handles the early exercise feature of American options, which is challenging for traditional numerical methods.
- **High Accuracy**: Provides accurate pricing of American options by learning the solution from the underlying PDE and boundary conditions.

## How It Works

1. **Problem Formulation**: The Black-Scholes PDE for American options is formulated, including the boundary conditions for early exercise.
2. **PINN Model Architecture**: A neural network is designed with multiple layers to approximate the solution of the PDE. The network is trained to minimize the PDE residual and satisfy the boundary conditions.
3. **Training Process**: The model is trained using a combination of collocation points (for PDE residuals), boundary conditions, and initial conditions. The loss function includes terms for the PDE residual, boundary conditions, and option values at expiration.
4. **Prediction**: Once trained, the PINN model can predict the option prices at any given point in time and stock price, including novel or unseen conditions.

## Implementation

The implementation involves the following components:

- **PINN Model**: A neural network model with several layers and activation functions to approximate the PDE solution.
- **Training Data**: Synthetic data generated for collocation points, boundary conditions, and initial conditions.
- **Training Process**: Optimization of the neural network parameters using the Adam optimizer and a loss function that combines the PDE residual and boundary conditions.
