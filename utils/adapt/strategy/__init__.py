# Import strategies.
from .adapt import AdaptiveParameterizedStrategy, ParameterizedStrategy
from .deepxplore import UncoveredRandomStrategy
from .dlfuzz import DLFuzzRoundRobin, MostCoveredStrategy
from .random import RandomStrategy

# Aliases for some strategies.
Adapt = AdaptiveParameterizedStrategy
DeepXplore = UncoveredRandomStrategy
DLFuzzFirst = MostCoveredStrategy
