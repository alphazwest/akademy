# Akademy 

Akademy is a module containing composable object classes for developing 
reinforcement learning algorithms focused on quantitative trading and 
time-series forecasting. This module is a work-in-progress and should, at no
time, be assumed to be designed well or be free of bugs.

# Overview
Akademy is designed using an `Agent`-`Environment` model such that `Agent`-class
objects ingest information from `Environment`-class objects (`Env`), produce
an `Action`, which is then applied to the `Environment` which results in a
change in `State` and possible reward to offer feedback to the agent.

*Note*: this module does not provide any training routines -- only the object class
that can be used to support the implementation of custom training routines.

# Getting Started

To install `akademy` use the following command in the desired Python 3.7+
environment:

`pip install akademy`

Once installed, developers will have access to `Agent`, `TradeEnv`, and `Network`
class objects in which to design Reinforcement Learning algorithms to train models.

Sample training routine:

```python
from akademy.models.envs import TradeEnv
from akademy.models.agents import DQNAgent
from akademy.common.utils import load_spy_daily

# loads the dataset used during training
data = load_spy_daily(count=2500)

# load the Trading Environment
env = TradeEnv(
    data=data,
    window=50,
    asset="spy",
)

# load the agent to train
agent = DQNAgent(
    env=env
)

# load user-defined training routine
training_routine(
    agent=agent,
    env=env
)
```

## Tests
Unit testing can be run via the following command:

`python -m unittest`

## Available Data
This module comes with minimal data for Agents and Environments to train on.
The current data available is listed below, along with sources for the most
up-to-date versions as well:

### 1. S&P500 
Location: `/data/SPY.CSV`\
Start:  `1993-01-29`\
End:    `2022-09-06`\
Total Rows: `7,454` (excludes header)\
Header: `Date,Open,High,Low,Close,Adj Close,Volume`\
Source: https://finance.yahoo.com/quote/SPY/history?p=SPY

*note*: Any data can be used easily enough via conversion into a Pandas DataFrame
object, but must contain information for `date` and pricing data for
`open`, `high`, `low`, and `close` as well as `volume` such that each row has
at least those 6 features or the latter 5 and an index representative of date.

# Notes

## Gym vs. Gymnasium
The `Gym` project by OpenAI has been sunset and now maintained as `Gymnasium` 
by the [Farama-Foundation](https://github.com/Farama-Foundation/Gymnasium). The
`Env` classes present here make use of the newer `Gymnasium` package which, among
other differences, produces an extra item in the `step` method indicating whether
an environment has been truncated. [See here](https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/core.py#L63)

## PyTorch
PyTorch requires some additional consideration for setup depending on use-case.
Akademy uses an approach whereby CPU-based training and inferences are possible
via parameterized function calls. However, GPU use (e.g. CUDA) requires local
considerations. [See here] (https://pytorch.org/get-started/locally/) for a more
in-depth discussion and guide.

This module currently uses the 1.* version, though a 2.* version release
is imminent and an upgrade to that version is planned.