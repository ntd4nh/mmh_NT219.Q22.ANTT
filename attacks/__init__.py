"""
Attacks Module
================
Implements both classical and DL-based attack methods for comparison.

Classical:
- CPA (Correlation Power Analysis)
- DPA (Differential Power Analysis)

DL-based:
- DLAttack (Log-likelihood accumulation with trained model)

Comparison:
- compare_attacks() — compare DL vs CPA on same data
"""

from attacks.classical import CPA, DPA
from attacks.dl_attack import DLAttack, compare_attacks
