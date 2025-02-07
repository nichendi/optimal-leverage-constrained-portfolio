# Project Overview

This repository includes the code implementation for the paper "Optimal Multi-period Leverage-Constrained Portfolios: A Neural Network Approach."

## Paper Abstract

We present a neural network approach for multi-period portfolio optimization that relaxes the long-only restriction and instead imposes a bound constraint on leverage.
We formulate the optimization problem for such a relaxed-constraint portfolio as a multi-period stochastic optimal control problem. We propose a novel relaxed-constraint neural network (RCNN) model to approximate the optimal control.
Using our proposed RCNN model transforms the original leverage-constrained optimization problem into an unconstrained one, which makes solving it computationally more feasible. We prove mathematically that the proposed RCNN control model can approximate the optimal relaxed-constraint strategy with arbitrary precision. We further propose to compute the optimal outperforming strategy over a benchmark based on cumulative quadratic shortfall (CS). Using U.S. historical market data from Jan 1926 to Jan 2023, we computationally compare and assess the proposed neural network approach to the optimal leverage-constrained strategy and long-only strategy respectively. We demonstrate that the leverage-constrained optimal strategy can achieve enhanced performance over the long-only strategy in outperforming a benchmark portfolio.

## Installation

First, clone this repository:

```bash
git clone git@github.com/nichendi/optimal-leverage-constrained-portfolio.git
cd optimal-leverage-constrained-portfolio
```

To install the required packages, create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Then, install the required dependecies:

```bash
pip install -r requirements.txt
```

## Project Structure

The key files are listed below:

optimal-leverage-constrained-portfolio/
├── config.py # Configuration file to store settings (e.g., data source, training parameters, model hyper-parameters, etc).
├── data/ # Directory to hold training data (tensors).
│ └── random_data.pt # A dummy data file (PyTorch tensor) included for demonstration.
├── models.py # Implements the RCNN model proposed in the paper.
├── objective.py # Implements the cumulative quadratic shortfall (CS) objective function.
├── train.py # Main script for model training.
└── wealth.py # Implements the wealth evolution function for calculating wealth trajectories.

## Usage

To train the model, run the following command:

```bash
python train.py
```

This will train the model on dummy data and save the trained parameters to `model.pt`.

Experiment settings can be modified in `config.py`.

## Data

In this paper, we use the real historical returns (1926-2023) for the equal-weighted and cap-weighted U.S. stock indexes, as well as the 10-year and 30-day Treasury indexes, adjusted for the Consumer Price Index (CPI).

The data was accessed through the Wharton Research Data Services (WRDS) platform, using an institutional license from the University of Waterloo, which restricts redistribution outside the authorized users.

To access this historical data from WRDS, please check with your institution to inquire about obtaining an institutional license.

However, the computational framework proposed in the paper is agnostic to the data source. For demonstration purposes, we have created a dummy random data file (`data/random_data.pt`). This file includes a tensor with dimensions of (1000, 4, 120), where 1000 is the number of paths, 4 is the number of assets, and 120 is the number of rebalancing periods (months).

## License

This project is licensed under the CC BY License.
