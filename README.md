
```
reinforcement-learning/
├── src/
│   ├── blackjack_experiment.py     # Blackjack VI and SARSA with variance analysis
│   ├── cartpole_experiment.py      # CartPole SARSA discretization sweep
├── results/                        # Experimental results (CSV files)
├── figures/                        # Generated plots and visualizations
├── report/
│   └── instructions/               # Assignment instructions
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```
## Installation

### 2. Set up Python Environment

#### **Linux/macOS:**
```bash
# Create virtual environment
python3 -m venv .venv

# Activate virtual environment
source .venv/bin/activate

# Install dependencies ORDER MATTERS HERE
pip install --upgrade pip
pip install -r requirements.txt
pip install git+https://github.com/jlm429/bettermdptools.git


# Verify installation
python --version
pip list

cd src
```

Make sure you are in the SRC folder when running these, they will run just move the folder structre around

**Blackjack Analysis:**
```bash
cd reinforcement-learning/src
cd src
python blackjack_experiment.py
```

**CartPole Discretization Study:**
```bash
cd reinforcement-learning/src
cd src
python cartpole_experiment.py
```


## Results

All experiments generate:
- **CSV files** in `results/` with detailed performance metrics
- **PNG figures** in `figures/` with visualizations and plots

### Key Metrics Tracked
- Convergence iterations and wall-clock time
- Episode rewards and lengths
- Policy visualizations
- Learning curves with statistical confidence
- Hyperparameter sensitivity analysis

