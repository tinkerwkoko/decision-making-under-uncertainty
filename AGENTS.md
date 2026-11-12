# AGENTS.md

**Authoritative project instruction and handover document.**

This file must be read by any AI coding agent before modifying this project.
It describes the project state, architecture, standards, and roadmap.

---

## 1. Project Name and Purpose

**Project:** Decision-Making Under Uncertainty: A Comparative Simulation and Evaluation Toolkit

**Purpose:** A research-oriented collection of implemented decision-making algorithms and reproducible experiments, PLUS a functional software engineering product (Streamlit application) that allows users to configure, run, observe, evaluate, and compare intelligent agents through a user-facing simulation interface — without manually editing Python source code.

The toolkit investigates how computational agents learn, optimise decisions, explore uncertain environments, and adapt to incomplete information.

Core research areas:
- Multi-armed bandits (exploration vs exploitation)
- Reinforcement learning (Q-learning)
- Dynamic programming (Value Iteration)
- Evolutionary computation (Genetic Algorithm)
- Partial observability and belief-state reasoning (POMDP)
- Transfer learning
- Experimental design and comparative evaluation

---

## 2. Project Vision and Final Deliverables

The final deliverable is BOTH:

1. **Research package** — verified implementations of completed algorithms, Q-learning evaluation, Value Iteration comparison, POMDP belief-state experiments, transfer-learning experiments, Genetic Algorithm implementation and comparison, automated test results, reproducible experiment configurations, research methodology, experimental results, graphs, interpretation, limitations, references, README, installation instructions.

2. **Streamlit application** — a functional user-facing application with laboratories for Bandits, GridWorld, POMDP, Transfer Learning, and Results/History. Users must be able to run experiments without editing code. Results must be visualised and exportable as CSV/JSON. Experiment configurations must be reproducible.

---

## 3. Research Questions

1. How do different exploration strategies (epsilon-greedy vs Thompson Sampling) affect cumulative regret in multi-armed bandits?
2. How does Q-learning (model-free, learns through interaction) compare with Value Iteration (model-based, requires known environment model) in GridWorld?
3. How does a Genetic Algorithm (population-based, evolutionary) compare with Q-learning (value-based, temporal-difference) for GridWorld policy optimisation?
4. How does an agent reason under partial observability using belief-state updates in a POMDP?
5. Can knowledge transferred from a source GridWorld task accelerate learning in a related target GridWorld task compared with learning from scratch?

---

## 4. Existing Architecture

```
decision-making-under-uncertainty/
├── algorithms/
│   ├── bandit_algorithms.py      # Thompson Sampling, Epsilon-Greedy
│   ├── q_learning.py             # Q-learning (train, choose_action, update, extract policy)
│   └── value_iteration.py        # Value Iteration (policy extraction, follow policy)
├── environments/
│   ├── bandit.py                 # BernoulliBandit environment
│   └── gridworld.py              # GridWorld environment
├── experiments/
│   ├── bandit_metrics.py         # Cumulative expected regret
│   ├── compare_bandit_algorithms.py  # Single-run bandit simulations
│   ├── evaluate_q_learning.py    # Q-learning evaluation script
│   ├── plot_bandit_comparison.py # Matplotlib plotting
│   ├── run_bandit_comparison.py  # Multi-simulation bandit comparison
│   ├── train_q_learning.py       # Q-learning training script
│   └── BANDIT_COMPARISON.md      # Bandit experiment documentation
├── tests/
│   ├── test_bandit.py
│   ├── test_bandit_algorithms.py
│   ├── test_bandit_metrics.py
│   ├── test_epsilon_greedy.py
│   ├── test_gridworld.py
│   ├── test_q_learning.py
│   ├── test_run_bandit_comparison.py
│   └── test_value_iteration.py
├── tests/__pycache__/
├── .venv/
├── README.md (empty)
├── requirements.txt (empty)
├── AGENTS.md (this file)
└── PROJECT_STATUS.md
```

### Separation of Concerns (existing)

- **algorithms/** — algorithm implementations only, no environment logic
- **environments/** — environment logic only (state transitions, rewards, termination)
- **experiments/** — experiment orchestration, metrics, plotting, evaluation
- **tests/** — unit tests for all components

---

## 5. Technology Stack

- **Language:** Python 3.14.7
- **Testing:** pytest 9.2.0
- **Plotting:** matplotlib
- **Randomness:** Python `random` module with `random.Random(seed)` for reproducibility
- **Virtual environment:** `.venv/` at project root
- **Planned:** Streamlit (for M6 user-facing application)
- **Planned:** SQLite or structured files (for M7 persistence/export)

No `pyproject.toml`, `setup.py`, `setup.cfg`, `conftest.py`, `pytest.ini`, or `tox.ini` exists.

---

## 6. Existing Working Implementations

All 87 existing tests pass (`python -m pytest -v`, 2.17s).

### Verified Components

| Component | File | Tests | Status |
|-----------|------|-------|--------|
| BernoulliBandit | `environments/bandit.py` | 10 | Working |
| Thompson Sampling | `algorithms/bandit_algorithms.py` | 14 | Working |
| Epsilon-Greedy | `algorithms/bandit_algorithms.py` | 13 | Working |
| Cumulative regret | `experiments/bandit_metrics.py` | 10 | Working |
| GridWorld | `environments/gridworld.py` | 11 | Working |
| Q-learning | `algorithms/q_learning.py` | 13 | Working |
| Value Iteration | `algorithms/value_iteration.py` | 8 | Working |
| Bandit comparison | `experiments/run_bandit_comparison.py` | 5 | Working |

### Verified Experiment Results

Running `run_comparison([0.1, 0.3, 0.5, 0.8], n_rounds=1000, n_simulations=100)`:
- Epsilon-Greedy final average cumulative regret: **54.276** ✓ (matches documentation)
- Thompson Sampling final average cumulative regret: **12.598** ✓ (matches documentation)

---

## 7. Algorithms and Environments

### Implemented

- **Bernoulli Bandit** — binary rewards, configurable probabilities, seeded RNG
- **Epsilon-Greedy** — epsilon-greedy arm selection, incremental average updates
- **Thompson Sampling** — Beta prior updates, posterior sampling for arm selection
- **Q-learning** — Q-table, epsilon-greedy action selection, TD update, terminal handling, epsilon decay, policy extraction
- **Value Iteration** — Bellman optimality updates, state-value computation, policy extraction, policy following
- **POMDP** — hidden states, observations, transition/observation probabilities, belief prediction, Bayesian belief updates, normalisation
- **Transfer learning** — Q-value transfer between source/target GridWorld tasks (4x4 grids, different goal)

### Not Yet Implemented (required by project vision)

- Genetic Algorithm for GridWorld
- Streamlit application with all laboratories
- Persistence and export (CSV/JSON)
- Experiment history

---

## 9. Scientific and Mathematical Assumptions (revised)

### Transfer Learning (M3) — DELIVERED AND DOCUMENTED
- Source task: 4x4 GridWorld, goal (3,3); Target task: 4x4 GridWorld, goal (3,2)
- State correspondence: Direct (same coordinates, same grid size)
- Transferred knowledge: Q-table from source initialises target Q-table
- Result: Transfer and from-scratch achieved identical performance (100% success, -5.0 avg reward, 5 steps)
- Interpretation: Transfer did not help in this specific task pair (small grid, simple policy, small goal difference). The result is reported honestly as a null transfer result.
- Streamlit application with all laboratories
- Persistence and export (CSV/JSON)
- Experiment history

---

## 8. Experimental Methodology

Every comparative experiment must define:
- Research question
- Hypothesis (where appropriate)
- Environment configuration
- Algorithm parameters
- Evaluation budget
- Number of independent runs
- Random seed strategy
- Metrics
- Results
- Limitations

Use consistent conditions where comparisons require them. Report variability, not only averages. Never fabricate results. Distinguish demonstrations, preliminary experiments, controlled comparisons, and scientifically supported conclusions.

Where algorithms have different information requirements or computational budgets, explain those differences rather than presenting an unfair comparison.

---

## 9. Scientific and Mathematical Assumptions

### Bandits
- Bernoulli reward distributions (binary 0/1 rewards)
- Stationary reward probabilities (do not change over time)
- Expected regret = sum over rounds of (best_probability - chosen_arm_probability)
- Distinguish expected regret from realised reward

### Q-learning
- Q-table update: Q(s,a) += alpha * (reward + gamma * max(Q(s',a')) - Q(s,a))
- Terminal states: future value = 0 when done=True
- Epsilon-greedy action selection with decay
- Model-free: learns through interaction, no environment model required

### Value Iteration
- Bellman optimality: V(s) = max_a sum over s' of P(s'|s,a) * (R(s,a,s') + gamma * V(s'))
- Assumes known environment model (transition probabilities and rewards)
- Converges when max change < theta
- Policy extraction: pi(s) = argmax_a of action values

### Comparison (Q-learning vs Value Iteration)
- Value Iteration requires full environment model; Q-learning does not
- Value Iteration computes optimal policy given full knowledge; Q-learning approximates through experience
- Do not present as identical information availability

### Genetic Algorithm (planned)
- Population-based policy optimisation
- Fitness = cumulative reward over episodes
- Selection, crossover, mutation operators
- No gradient information required

### POMDP (planned)
- Hidden states, observations, transition probabilities, observation probabilities, rewards
- Bayesian belief update: b'(s') ∝ O(o|s',a) * sum over s of T(s'|s,a) * b(s)
- Normalisation to maintain valid probability distributions

### Transfer Learning (planned)
- Related GridWorld environments with state correspondence
- Transferred knowledge: Q-values or policy from source task
- Compare learning from scratch vs transfer

---

## 10. Coding Standards

- Follow existing code conventions in the repository
- Use type hints where consistent with existing style
- Functions must have docstrings explaining purpose, args, returns
- Input validation with `ValueError` for invalid parameters
- Use `random.Random(seed)` for reproducibility; never use bare `random` module functions in experiment code without a seeded RNG
- No silent omission of features
- No placeholder implementations claimed as complete
- Do not refactor unrelated modules
- One logical module + its tests per implementation step
- No more than 3 files modified in a single step without approval

### Naming conventions
- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_SNAKE_CASE`

### Import style
- Relative imports within package (e.g., `from algorithms.q_learning import ...`)
- No `sys.path.insert` hacks in new code

---

## 11. Testing Requirements

- Run `python -m pytest -v` after every milestone
- Report actual test output: passed, failed, skipped
- Preserve existing tests; do not remove or weaken them
- Add tests for all new functionality:
  - Algorithm correctness
  - Environment behaviour
  - Invalid inputs
  - Random-seed reproducibility
  - Numerical edge cases
  - Genetic operators
  - POMDP belief updates
  - Transfer compatibility
  - Experiment orchestration
  - Data export and persistence
- If a test must change because intended behaviour legitimately changed, explain why and document the change

---

## 12. Randomness and Reproducibility Requirements

- All experiments must use seeded RNG: `random.Random(seed)`
- Same seed must produce identical results across runs
- Bandit comparison uses `seed = simulation` index for each independent simulation
- Q-learning training accepts `seed` parameter
- Reproducibility must be verified by running experiments twice with same seed and comparing results
- Document seed strategy for every experiment

---

## 13. Documentation Standards

Required documentation files:
- `README.md` — installation, setup, application startup, running experiments
- `AGENTS.md` — this file (authoritative agent handover)
- `PROJECT_STATUS.md` — milestones, tasks, test results, known issues, next task
- `docs/RESEARCH_METHODOLOGY.md` — research questions, hypotheses, experimental design
- `docs/ALGORITHM_EXPLANATIONS.md` — mathematical foundations, equations, how each algorithm works
- `docs/EXPERIMENT_RESULTS.md` — results, plots, interpretation, limitations
- `docs/REFERENCES.md` — cited algorithms and research sources (verify references exist)
- `docs/ARCHITECTURE.md` — system architecture, component interactions
- `docs/USER_GUIDE.md` — how to use the Streamlit application

Cite established algorithms and original research sources. Do not claim established algorithms as original inventions. Distinguish published methods from implementation choices and experimental contributions.

---

## 14. Attribution and Academic Integrity

- Do not claim established algorithms as original inventions
- Cite established algorithms and original research sources
- Verify that cited references actually exist and support claims
- Clearly distinguish existing published methods from implementation choices and experimental contributions
- Do not fabricate results
- Do not interpret a single experimental result as universal evidence
- State what each experiment can and cannot establish

---

## 15. User-Interface Requirements (Streamlit, M6)

- Use Streamlit unless a compelling technical reason exists to choose another framework
- Sections: Home/Overview, Experiment Configuration, Bandit Laboratory, GridWorld Laboratory, POMDP Laboratory, Transfer Learning Laboratory, Results/Experiment History
- Users must be able to: select environment, select algorithm, configure parameters, set simulation length, set independent runs, specify seeds, validate inputs
- Only display parameters relevant to selected algorithm
- Every visible control must work (no decorative buttons or fake functionality)
- Handle errors and invalid inputs clearly
- UI calls same tested experiment functions available to command-line users
- Support both programmatic execution and UI-driven execution
- No network access required for ordinary experiments
- Export experiment data as CSV and JSON

---

## 16. Complete Development Roadmap (Milestones)

| Milestone | Focus | Status |
|-----------|-------|--------|
| M0 | Repository audit, AGENTS.md, PROJECT_STATUS.md, audit presentation | DONE |
| M1 | Q-learning and Value Iteration evaluation, tests, visualisations, documentation | DONE |
| M2 | POMDP and belief-state demonstration, Bayesian updates, tests | DONE |
| M3 | Transfer learning experiment, source/target GridWorld, comparison | DONE |
| M4 | Genetic Algorithm, GridWorld evaluation, Q-learning comparison | PENDING |
| — | Submission-ready research package (after M4, before M5) | PENDING |
| M5 | Reusable simulation engine, experiment infrastructure | PENDING |
| M6 | Streamlit application (laboratories one at a time) | PENDING |
| M7 | Persistence, export, documentation, final integration | PENDING |

Do not start a milestone until the previous one is reviewed and approved by the user.

---

## 17. Completed Work

- Repository structure established with algorithms/, environments/, experiments/, tests/
- Bernoulli Bandit environment implemented and tested
- Epsilon-Greedy and Thompson Sampling implemented and tested
- Cumulative expected regret metric implemented and tested
- Q-learning implemented and tested (with epsilon decay, terminal handling, policy extraction)
- Value Iteration implemented and tested (Bellman updates, policy extraction, following)
- GridWorld environment implemented and tested
- Bandit comparison experiment (multi-simulation) implemented and tested
- Q-learning evaluation experiments (multi-run, epsilon=0.1 and 0.3, success rates, learning curves)
- Value Iteration evaluation (convergence, policy extraction, path following)
- Q-learning vs Value Iteration comparison (documented information requirement differences)
- 156 tests, all passing (87 original + 69 new across M1-M3)
- Verified bandit results: Epsilon-Greedy 54.276, Thompson Sampling 12.598
- POMDP environment with hidden states, observations, transitions, observations, rewards
- POMDP belief prediction, Bayesian belief updates, normalisation
- POMDP belief-state demonstration with step-by-step belief updates
- POMDP tests for valid belief distributions, prediction, Bayesian updates, edge cases, invalid inputs
- Transfer learning experiment: Q-value transfer between source and target GridWorld tasks
  - Source: 4x4 GridWorld, goal (3,3); Target: 4x4 GridWorld, goal (3,2)
  - Results: Transfer and from-scratch achieved identical performance (100% success, -5.0 avg reward, 5 steps)
  - Interpretation documented honestly as a null transfer result

---

## 18. In-Progress Work

- M3: Transfer learning experiment — complete, awaiting user approval

---

## 19. Outstanding Work

- M4: Genetic Algorithm (population, fitness, selection, crossover, mutation, replacement)
- M4: GA vs Q-learning comparison
- M4: Submission-ready research package
- M5: Reusable simulation engine, experiment infrastructure
- M6: Streamlit application (all laboratories)
- M7: Persistence, export (CSV/JSON), experiment history, final integration
- Documentation: README.md, docs/*.md files
- All remaining tests for new functionality

---

## 20. Known Bugs and Limitations

- `evaluate_q_learning.py` and `train_q_learning.py` use `sys.path.insert` hacks — should be cleaned up when restructuring
- `plot_bandit_comparison.py` uses relative save path and `plt.show()` — may block in non-interactive environments
- `plot_pomdp_belief.py` calls `plt.show()` which may block in non-interactive environments (same issue)
- `plot_transfer_learning.py` also calls `plt.show()` — may block in non-interactive environments
- No Genetic Algorithm or Streamlit implementation exists yet
- No persistence or export mechanism exists yet
- No `requirements.txt` contents (empty file)
- No `README.md` contents (empty file)
- No `docs/` directory exists

---

## 21. Important Architectural Decisions

- **Preserve existing working code** — do not delete working features, tests, or results without documented reason
- **Do not rewrite architecture simply due to preference** — build upon existing structure
- **Separate environment logic from algorithm implementations** — already established in existing code
- **UI must call same tested experiment functions as command-line** — experiment orchestration separated from UI
- **Use `random.Random(seed)` for all randomness** — ensures reproducibility
- **Streamlit for UI** — unless compelling technical reason emerges during audit

---

## 22. Commands for Installation, Testing, and Execution

```bash
# Install dependencies
pip install -r requirements.txt  # requirements.txt is currently empty; add streamlit, matplotlib, pytest

# Run all tests
python -m pytest -v

# Run bandit comparison
python -c "from experiments.run_bandit_comparison import run_comparison; results = run_comparison([0.1, 0.3, 0.5, 0.8], n_rounds=1000, n_simulations=100)"

# Train Q-learning
python experiments/train_q_learning.py

# Evaluate Q-learning
python experiments/evaluate_q_learning.py
```

---

## 23. Learning and Technical Explanation Requirements

For every major algorithm and research component:
- Explain underlying concepts before or alongside implementation
- Explain mathematical foundations and equations
- Explain purpose of important functions
- Explain important implementation choices
- Explain connection to research questions
- Identify assumptions, limitations, potential sources of error
- Provide practical examples and walkthroughs
- Explain how tests validate the implementation
- Identify questions the user should be able to answer

Target audience: Computer Science undergraduate developing knowledge of reinforcement learning, computational neuroscience, and decision-making under uncertainty. Do not assume advanced mathematics or RL terminology. Do not overwhelm with unexplained code dumps.

At the end of each milestone, provide a technical learning guide covering:
1. What was built
2. How it works
3. Why the approach was chosen
4. The mathematics behind it
5. How components interact
6. What the tests establish
7. What the user should study
8. Questions the user should answer independently

---

## 24. Completion Criteria (All Must Be Addressed)

1. Existing algorithms verified ✓ (87 tests pass)
2. Q-learning evaluation complete
3. Value Iteration verified and compared
4. Genetic Algorithm implemented and tested
5. POMDP belief-state demonstration implemented and tested
6. Transfer learning experiment implemented and evaluated
7. Reusable simulation engine functional
8. User-facing application functional
9. All algorithms accessible through interface
10. Results visualised and exported
11. Experiment configurations reproducible
12. Tests pass with exceptions documented
13. Documentation complete
14. AGENTS.md and PROJECT_STATUS.md accurate
15. Final integration review
16. Submission-ready research package complete
17. Technical learning guides provided

Each criterion in PROJECT_STATUS.md must have evidence: DONE (with specific test/output), NOT DONE (what remains), or BLOCKED (issue and decision required).

---

**End of AGENTS.md.** This document must be kept updated whenever meaningful project decisions, implementations, or architectural changes occur.