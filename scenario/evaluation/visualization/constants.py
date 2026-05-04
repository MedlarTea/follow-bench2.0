from __future__ import annotations

import os


EVALUATION_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_LOG_ROOT = os.path.join(EVALUATION_DIR, "logs")
DEFAULT_RESULTS_DIR = os.path.join(EVALUATION_DIR, "results")

METHOD_LABELS = {
    "sfm": "SFM",
    "dwa_traj": "DWA",
    "rda": "MPC",
    "rda_traj": "MPC w/ Traj.",
    "rda_search": "MPC + DS",
    "bso_hfc": "BSO-HFC",
    "trackvla": "TrackVLA",
    "oa_vat": "OA-VAT",
}

COLORS = {
    "sfm": "#54a24b",
    "dwa_traj": "#f58518",
    "rda": "#e45756",
    "rda_traj": "#9d755d",
    "rda_search": "#72b7b2",
    "bso_hfc": "#13a8a8",
    "trackvla": "#4c78a8",
    "oa_vat": "#b279a2",
}

LINE_STYLES = {
    "back": "-",
    "left_side": "--",
    "right_side": ":",
}

PLOTTED_METRICS = {
    "SR": {"unit": "%", "kind": "percent"},
    "ASR": {"unit": "%", "kind": "percent"},
    "TVR": {"unit": "%", "kind": "percent"},
    "TinTPerson": {"unit": "%", "kind": "percent"},
    "TinPrivate": {"unit": "%", "kind": "percent"},
    "PL": {"unit": "m", "kind": "scalar"},
    "AvgVel": {"unit": "m/s", "kind": "scalar"},
    "AvgAcc": {"unit": "m/s^2", "kind": "scalar"},
    "Jerk": {"unit": "m/s^3", "kind": "scalar"},
    "CostMean": {"unit": "ms", "kind": "scalar"},
    "CostP95": {"unit": "ms", "kind": "scalar"},
    "PolicyMean": {"unit": "ms", "kind": "scalar"},
    "PolicyP95": {"unit": "ms", "kind": "scalar"},
    "PlanMean": {"unit": "ms", "kind": "scalar"},
    "PlanP95": {"unit": "ms", "kind": "scalar"},
    "PercMean": {"unit": "ms", "kind": "scalar"},
    "PercP95": {"unit": "ms", "kind": "scalar"},
    "DetMean": {"unit": "ms", "kind": "scalar"},
    "TrackMean": {"unit": "ms", "kind": "scalar"},
    "ReIDMean": {"unit": "ms", "kind": "scalar"},
    "MapMean": {"unit": "ms", "kind": "scalar"},
    "FSMMean": {"unit": "ms", "kind": "scalar"},
}

DEFAULT_PLOT_METRICS = ["SR", "ASR", "TVR", "TinTPerson", "TinPrivate", "Jerk"]
DEFAULT_TABLE_METRICS = [
    "SR",
    "ASR",
    "TVR",
    "TinTPerson",
    "TinPrivate",
    "PL",
    "AvgVel",
    "AvgAcc",
    "Jerk",
    "PolicyMean",
    "PolicyP95",
    "PlanMean",
    "PlanP95",
    "PercMean",
    "PercP95",
]
