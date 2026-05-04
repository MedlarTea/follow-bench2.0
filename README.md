# Follow-Bench 2.0 (Anonymous Code Release)

A benchmark for **socially-aware robot person following (RPF)** in photorealistic
simulation. This anonymous release contains the full evaluation framework, all
planners (model-based + learning-based), perception/ReID stack, and one of our
four scenarios (`random`). Scenario assets, simulator binaries, and pretrained
weights are subject to institutional release approval and will be made public
in the camera-ready version.

> **Anonymity notice (NeurIPS double-blind):**
> Identifying information has been removed. Vendored third-party code under
> `scenario/planners/learning_based/oa-vat/{ORTrack,dinov3-main}/` and
> `scenario/target_identification/reid_kpr/deep_person_reid/` retains its
> original (third-party, public) attribution.

---

## 1. Directory layout

```
followbench2.0-light/
├── README.md                           # this file (only doc in the release)
├── environment.yml                     # conda env (Python 3.10, CARLA 0.9.16)
├── .gitignore
└── scenario/
    ├── debug_vis/                      # 2D top-down visualiser (live overlay)
    ├── evaluation/                     # metrics, schemas, collision logging
    │   ├── core/                       #   logger, collision monitor, scoring
    │   └── visualization/              #   replay + plot utilities
    ├── map_annotator/                  # nav-mesh + ROI flow-point editor
    ├── planners/                       # all follower policies + perception
    │   ├── adapters/                   #   planner adapters (FollowerPolicyAdapter)
    │   ├── behavior/                   #   side-follow, search-state machinery
    │   ├── common/                     #   maps, prediction, utilities
    │   ├── learning_based/
    │   │   ├── trackvla/               #   end-to-end VLA baseline (re-impl.)
    │   │   └── oa-vat/                 #   YOLOe + ORTrack + DINOv3 ReID baseline (re-impl.)
    │   ├── perception/                 #   GT / sensor (RGB-D + LiDAR) frontends
    │   ├── planning/                   #   PID, SFM, DWA, RDA, BSO-HFC kernels
    │   ├── prediction/                 #   trajectory predictors (CVKF, S-GAN, …)
    │   ├── tests/                      #   unit / smoke tests
    │   ├── traj_predictor/             #   trained checkpoints loader
    │   └── vendors/                    #   vendored RDA-planner (do not modify)
    ├── target_identification/          # appearance ReID (basic + KPR/SOLIDER)
    │   ├── reid_kpr/                   #   vendored deep_person_reid + KPR
    │   ├── reid_model/                 #   our ResNet-based extractor
    │   ├── states/                     #   FSM (Initial → Tracking → Reid)
    │   └── reid_classifier.py          #   online Ridge confidence head
    ├── random/                         # the only scenario in this release
    │   ├── runner/                     #   episode manager (shared main loop)
    │   ├── core_types.py               #   FollowObservation / RobotState / NpcState
    │   ├── robot_runtime.py            #   robot spawn + sensor harness
    │   ├── pedestrian_sfm.py           #   social-force NPC controller
    │   ├── visibility_instance.py      #   instance-mask visibility check
    │   ├── carla_roi_crowd_runner.py   #   crowd flow inside ROI
    │   └── run_episode_manager.sh      #   one-line launcher
    └── weight_paths.py                 # repo-relative weight resolution helper
```

The other three scenarios (`corridor`, `doorway`, `clutter`) share the same
manager/runtime under `scenario/random/runner/`; their assets and launch
scripts are withheld for the anonymous review and released in the
camera-ready version.

---

## 2. Setup

### 2.1 Simulator

We use **CARLA 0.9.16** with a custom NavMesh + crowd-flow extension. The
custom build, walker assets, and pretrained weights are part of the
**institutional release** — we will publish them alongside the camera-ready
paper. For review, only the code in this repository is available.

If you have access to a stock CARLA 0.9.16 install, the framework imports and
unit tests will run; full-scenario simulation additionally needs our
walker / NavMesh assets.

### 2.2 Python environment

```bash
conda env create -f environment.yml
conda activate followbench

# KPR ReID needs two source-only packages:
pip install "segment_anything @ git+https://github.com/facebookresearch/segment-anything"
pip install "cosine_annealing_warmup @ git+https://github.com/katsura-jp/pytorch-cosine-annealing-with-warmup"
```

### 2.3 Weights layout

Place pretrained weights under `data/weights/` (kept out of this release):

```
data/
├── weights/
│   ├── yolo/                yolo11s.pt, yoloe-11l-seg.pt
│   ├── reid_basic/          ckpt.t7
│   ├── reid_kpr/            kpr_*.pth.tar
│   └── traj_predictor/      sgan.pt, csgan.pt
├── trackvla/                checkpoint-* (Qwen3-4B + SigLIP + DINOv3)
└── oa-vat/
    ├── dinov3/              dinov3_vit{b16,s16}_*.pth
    ├── ortrack/             ORTrack_ep0300.pth.tar
    ├── yolo/                yoloe-11l-seg.pt
    └── mobileclip_blt.ts
```

`scenario/weight_paths.py` resolves all paths relative to the repo root, so no
path editing is required.

---

## 3. Quick start

### 3.1 Run a demo episode (random scenario)

```bash
cd followbench2.0-light
PLANNER=pid FOLLOW_POSITION=back ./scenario/random/run_episode_manager.sh
```

Useful environment overrides (consumed by `run_episode_manager.sh`):

| Var                | Default | Purpose                                           |
|--------------------|---------|---------------------------------------------------|
| `PLANNER`          | `pid`   | follower policy (see §4 table)                    |
| `FOLLOW_POSITION`  | `back`  | `back`, `left_side`, `right_side`                 |
| `DESIRED_DISTANCE` | `1.5`   | metres between robot and target                   |

Pass any other CLI flag through; e.g. enable the perception frontend that
decouples planners from GT:

```bash
PLANNER=rda ./scenario/random/run_episode_manager.sh \
    --use-perception --reid-mode kpr
```

### 3.2 Direct Python entry-point

```bash
conda run -n followbench python scenario/random/runner/run_episode_manager.py --help
```

`build_parser()` at the top of `run_episode_manager.py` is the authoritative
list of every CLI flag.

### 3.3 Evaluation

After one or more episodes finish, compute aggregated metrics:

```bash
conda run -n followbench python scenario/evaluation/core/score.py \
    --runs-glob "scenario/random/runs/2026*/episode.json" \
    --out scenario/evaluation/results/random_summary.csv
```

Schemas, collision logic, and scoring are in `scenario/evaluation/core/`.

### 3.4 Live debug visualiser

Add `--debug` to any `run_episode_manager.py` call to launch the 2D top-down
visualiser in a side process. Source under `scenario/debug_vis/`.

---

## 4. Available planners and follow settings

Follow-Bench 2.0 ships 12 planners covering classical model-based control,
sensor-driven perception variants, and end-to-end learned policies. All
planners implement the same `FollowerPolicyAdapter` ABC
(`scenario/random/follow_policy_adapter.py`) with `reset()` and `act()`.

| Planner              | Type        | Perception input           | Tracking / Re-acquisition       | Follow position | CLI                                     |
|----------------------|-------------|----------------------------|---------------------------------|-----------------|-----------------------------------------|
| `pid`                | Classical   | GT pose                    | —                               | back / side     | `--planner pid`                         |
| `sfm`                | Classical   | GT pose                    | —                               | back / side     | `--planner sfm`                         |
| `dwa_traj`           | Classical   | GT pose + traj-pred        | —                               | back / side     | `--planner dwa_traj`                    |
| `dwa_traj_depth_tpt` | Classical   | RGB-D + traj-pred          | Appearance ReID (basic/KPR)     | back / side     | `--planner dwa_traj_depth_tpt`          |
| `rda`                | Classical   | GT pose                    | —                               | back / side     | `--planner rda`                         |
| `rda_lidar`          | Classical   | LiDAR                      | Geometry-only                   | back / side     | `--planner rda_lidar`                   |
| `rda_traj`           | Classical   | GT pose + traj-pred        | —                               | back / side     | `--planner rda_traj`                    |
| `rda_search`         | Classical   | RGB-D + ReID               | Appearance + geometry recovery  | back / side     | `--planner rda_search`                  |
| `rda_depth_tpt`      | Classical   | RGB-D + ReID               | Appearance ReID (basic/KPR)     | back / side     | `--planner rda_depth_tpt`               |
| `bso_hfc`            | Classical   | GT pose                    | Hierarchical fuzzy controller   | back / side     | `--planner bso_hfc`                     |
| `trackvla` *         | Learned     | RGB (front)                | VLA, ego-centric prompts        | back            | `--planner trackvla`                    |
| `oa_vat` *           | Learned     | RGB (front)                | YOLOe → ORTrack → DINOv3 ReID + PID | back        | `--planner oa_vat`                      |

\* Re-implemented from public baselines (TrackVLA [1], OA-VAT [2]) under the
shared `FollowerPolicyAdapter` interface for fair comparison.

**Perception frontend.** Any classical planner above can be wrapped with a
multi-view YOLO + depth + ReID frontend that replaces GT poses with detected
tracks:

```bash
--use-perception --reid-mode {basic,kpr}
```

`basic` is a lightweight ResNet extractor; `kpr` is the SOLIDER-Swin
keypoint-promptable ReID (occlusion-robust). The end-to-end planners
(`trackvla`, `oa_vat`) carry their own perception and ignore this flag.

**Follow-position policies.** `back` (classical PETS-style), `left_side`,
`right_side` (socially-aware abreast follow). The target-route lane bias has
two modes:

```bash
--target-lane-bias-mode {right_hand, leave_follow_side_clear}
```

---

## 5. Comparison with prior benchmarks

We position Follow-Bench 2.0 along three capability axes required by
socially-aware RPF: **target re-identification**, **obstacle / occlusion
avoidance**, and **socially-aware following**.

<!-- markdown rendering of the comparison table; LaTeX source kept below -->

| Benchmark                             | ReID | Avoid. | Social | Follow Conf. | MB | LB | Eval. type        | Ped. interaction                          | Engine            |
|---------------------------------------|:----:|:------:|:------:|--------------|:--:|:--:|-------------------|-------------------------------------------|-------------------|
| EVT-Bench [1]                         | ++   | ++     | +      | Back         |    | ✓  | task-level        | ORCA*                                     | Habitat 3.0       |
| Gym-UnrealCV [3]                      | +    | +      | +      | Back         |    | ✓  | task-level        | NavMesh                                   | Unreal Engine     |
| DAT (aerial) [4]                      | +    | +      | +      | Back         |    | ✓  | task-level        | NavMesh                                   | Unreal Engine     |
| TPT-Bench [5]                         | +++  | —      | —      | —            | —  | —  | perception-level  | real traj. (offline)                      | real-world seq.   |
| Follow-Bench 1.0 [6]                  | —    | +++    | ++     | Back+Side    | ✓  |    | planning-level    | SFM / ORCA                                | 2D simulator      |
| **Follow-Bench 2.0 (ours)**           | ++   | +++    | +++    | Back+Side    | ✓  | ✓  | task-level        | NavMesh + SFM/ORCA + social activities    | Unreal Engine     |

`+`, `++`, `+++` = weak / moderate / strong coverage; `—` = outside the
benchmark's task definition. MB = model-based planner; LB = learning-based
planner. ORCA* indicates dense mesh-agent interactions may still allow
penetration in crowded cases.

> The LaTeX source for the camera-ready version of this table is preserved at
> the bottom of this file (§Appendix A).

---

## 6. Scenario showcase

Follow-Bench 2.0 contains four scenarios; only `random` is in this code
release, but representative recordings of all four are linked below.
Videos / GIFs will be uploaded to an anonymous host before the rebuttal
period.

<table>
  <tr>
    <td align="center" width="25%"><b>Corridor</b><br/><i>placeholder for video — to be uploaded</i></td>
    <td align="center" width="25%"><b>Doorway</b><br/><i>placeholder for video — to be uploaded</i></td>
    <td align="center" width="25%"><b>Clutter</b><br/><i>placeholder for video — to be uploaded</i></td>
    <td align="center" width="25%"><b>Random (released)</b><br/><i>placeholder for video — to be uploaded</i></td>
  </tr>
  <tr>
    <td align="center"><img src="docs/scenario_corridor.png" alt="corridor" width="220"/></td>
    <td align="center"><img src="docs/scenario_doorway.png" alt="doorway" width="220"/></td>
    <td align="center"><img src="docs/scenario_clutter.png" alt="clutter" width="220"/></td>
    <td align="center"><img src="docs/scenario_random.png" alt="random" width="220"/></td>
  </tr>
</table>

*Image files referenced above will be added to `docs/` before the rebuttal.*

---

## 7. Architectural notes

* Every planner inherits from `FollowerPolicyAdapter` (`reset() / act()`).
* New planners are registered in
  `scenario/random/runner/run_episode_manager.py` under the `--planner`
  argparse choices.
* Planners that expose debug info to the visualiser implement
  `get_debug_info() -> dict` returning `obstacles` and `traj_points`.
* Files under `scenario/planners/vendors/` are vendored third-party code
  (RDA-planner) and are never modified in-tree.
* The 2D debug visualiser in `scenario/debug_vis/` runs in a separate
  process and is non-blocking by design.

---

## References

[1] *EVT-Bench / TrackVLA*: Wang et al., *TrackVLA: A Vision-Language-Action
Model for Embodied Visual Tracking*, 2025.
[2] *OA-VAT*: Anonymous, CVPR 2026 (re-implemented baseline).
[3] *Gym-UnrealCV*: Qiu et al., *UnrealCV*, 2017.
[4] *DAT (aerial)*: Sun et al., open-source release.
[5] *TPT-Bench*: Ye et al., *TPT-Bench*, 2025.
[6] *Follow-Bench 1.0*: Ye et al., *RPF*, 2025.

---

## Appendix A — LaTeX source for the comparison table

```latex
\begin{table}[t]
  \centering
  \caption{Comparison of \textbf{Follow-Bench~2.0} with representative benchmarks for embodied visual tracking (EVT), target-person tracking (TPT), and robot person following (RPF). We characterize each benchmark along the three capabilities required by socially-aware RPF: target re-identification, obstacle / occlusion avoidance, and socially-aware following. Symbols $+$, $++$, and $+++$ indicate weak, moderate, and strong coverage; ``--'' marks axes that are outside the benchmark's task definition. MB and LB denote model-based and learning-based planners. ORCA* indicates that EVT-Bench reports ORCA-based avoidance, while dense mesh-agent interactions may still allow penetration in crowded cases.}
  \label{tab:bench-comparison}
  \renewcommand{\arraystretch}{1.5}
  \setlength{\tabcolsep}{10pt}
  \resizebox{\linewidth}{!}{%
  \begin{tabular}{l c c c l c c l l l}
  \toprule
  \multirow{2}{*}{\textbf{Benchmark}}
    & \multicolumn{3}{c}{\textbf{RPF Capability Challenge}}
    & \multirow{2}{*}{\textbf{Follow Conf.}}
    & \multicolumn{2}{c}{\textbf{Eval. Planners}}
    & \multirow{2}{*}{\textbf{Eval. Type}}
    & \multirow{2}{*}{\textbf{Ped. Inter.}}
    & \multirow{2}{*}{\textbf{Engine}} \\
  \cline{2-4}\cline{6-7}
  \noalign{\vskip 2.5pt}
    & \makecell[c]{Target\\ReID}
    & \makecell[c]{Obstacle /\\Occlusion Avoid.}
    & \makecell[c]{Socially-Aware\\Following}
    & & MB & LB & & & \\
  \midrule
  \multicolumn{10}{l}{\emph{Embodied visual-tracking benchmarks: back-following only; perception entangled inside the policy; comfort largely ignored.}}\\
  \midrule
  EVT-Bench~\cite{wang2025trackvla}
    & ++ & ++ & + & Back & \xmark & \cmark
    & task-level & ORCA* & Habitat\,3.0 \\
  Gym-UnrealCV~\cite{qiu2017unrealcv}
    & + & + & + & Back & \xmark & \cmark
    & task-level & NavMesh & Unreal Engine \\
  DAT (aerial)~\cite{sunopen}
    & + & + & + & Back & \xmark & \cmark
    & task-level & NavMesh & Unreal Engine \\
  \midrule
  \multicolumn{10}{l}{\emph{RPF-related benchmarks: the task is RPF, but perception and planning are not jointly evaluated.}}\\
  \midrule
  TPT-Bench~\cite{ye2025tpt}
    & +++ & -- & -- & -- & -- & --
    & perception-level & \makecell[l]{real traj.\\(offline)} & \emph{Real-world seq.} \\
  Follow-Bench\,1.0~\cite{ye2025rpf}
    & -- & +++ & ++ & Back\,+\,Side & \cmark & \xmark
    & planning-level & SFM\,/\,ORCA & 2D Simulator \\
  \textbf{Follow-Bench\,2.0}
    & ++ & +++ & +++ & Back\,+\,Side & \cmark & \cmark
    & task-level
    & \makecell[l]{NavMesh +\\SFM/ORCA +\\social activities}
    & Unreal Engine \\
  \bottomrule
  \end{tabular}%
  }
\end{table}
```
