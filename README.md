# BayesianExoAdaptation

The whole project tests one causal chain, and every experiment below exists to validate a specific link in it:

1. A neural network trained to MAP on source data only becomes overconfident under domain shift (calibration breaks on target data). This is a known failure mode of deep learning models, and it is particularly problematic in safety-critical applications like assistive robotics.
2. A post-hoc Bayesian treatment (Kronecker-Factored Laplace Approximation, KFLA) yields a predictive epistemic variance that grows systematically as samples move farther from the source distribution (variance acts as a shift detector).
3. That same variance signal can drive adaptation: down-weighting high-uncertainty samples during entropy-based adaptation lets the optimization be steered mainly by reliable target samples.

We want to apply Source-Free Domain Adaptation (SFDA) in assistive robotics, specifically for power-assist exoskeletons. The goal is to adapt a myoelectric (EMG) controller, pre-trained on a pool of healthy subjects (Source Domain), to a new, unseen patient (Target Domain) without accessing the original training data.

## Dataset: Lower Limb Surface Electromyography (sEMG)

### Overview
This project uses a surface electromyography (sEMG) database designed for lower limb analysis. The dataset contains recordings from **22 male subjects**: 11 with previously diagnosed knee abnormalities and 11 healthy controls.

### Acquisition Protocol
Each subject performed **3 exercises** to analyze muscular behavior related to knee movement:
1. **March** - Walking motion
2. **Leg Extension** - Extension from seated position
3. **Knee Flexion** - Flexion while standing

Each exercise set contains 3-5 repetitions.

### Electrode Placement
Four electrodes were placed on the following muscles of the leg:

| Channel | Muscle | Abbreviation | Column Index |
|---------|--------|--------------|--------------|
| Ch1 | Rectus Femoris | RF | 0 |
| Ch2 | Biceps Femoris | BF | 1 |
| Ch3 | Vastus Medialis (Internus) | VM | 2 |
| Ch4 | Semitendinosus | ST | 3 |
| Ch5 | Knee Flexion Angle | FX | 4 |

### Data Format
Each file contains 5 columns with the following specifications:

| Channel | Description | Units | Sample Count |
|---------|-------------|-------|--------------|
| RF, BF, VM, ST | EMG signals | mV | ~15,300 values |
| FX | Knee flexion angle | degrees | ~765 values (extrapolated from 50 to 1000 Hz) |

### File Organization
The dataset is organized into 2 folders:
- `A_TXT` - Abnormal subjects, .txt format
- `N_TXT` - Normal subjects, .txt format

### Classes
- **Normal (N)**: 11 healthy subjects
- **Abnormal (A)**: 11 subjects with knee pathologies
