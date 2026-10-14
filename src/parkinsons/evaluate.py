"""Inputs to ``skore.evaluate`` for the 01_dummy experiment.

Uses a simple 80/20 random row holdout (``splitter=0.2``), matching the
lab guide's specification for the dummy and ridge baselines. The patient-
level grouping leak is intentional at this stage; it will be addressed when
the grouped-CV experiments begin.
"""

from __future__ import annotations

# 80/20 random holdout — fixed for 01_dummy and 02_ridge per the lab guide.
# skore interprets a float in (0, 1) as the test fraction.
splitter = 0.2
