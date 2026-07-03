# AI2Pot-cli

Copyright © 2025 Hanyu Liu.

AI2Pot-cli is distributed under the GNU General Public License v3.0.

## What is AI2Pot-cli?

**AI2Pot-cli** is the official command-line interface for **AI2Pot** (https://github.com/lhycms/AI2Pot). It provides an interactive and scriptable toolkit for dataset preprocessing, training-input generation, model evaluation, post-processing, and molecular-dynamics deployment.

## Installation

AI2Pot-cli can be installed from source.

```bash
$ git clone https://github.com/lhycms/AI2Pot-cli.git
$ cd AI2Pot-cli
$ pip install .
```

After installation, the command `ai2pot-cli` should be available in the current Python environment.

```bash
$ ai2pot-cli --version
```

## Usage

AI2Pot-cli supports both an interactive menu interface and command-line execution.

### 1. Interactive mode

Run:

```bash
$ ai2pot-cli
```

The following menu will be shown:

```shell
 +--------------------------------------------------------------------------+
 |                       AI2Pot-cli Standard Edition                        |
 |                           Version 1.0.0                                  |
 |                                                                          |
 |                Official Command Line Interface for AI2Pot                |
 |                                                                          |
 |               Developer: Hanyu Liu (hyliu2016@buaa.edu.cn)               |
 |                                                                          |
 |            AI2Pot-cli : https://github.com/lhycms/AI2Pot-cli             |
 |                AI2Pot : https://github.com/lhycms/AI2Pot                 |
 +--------------------------------------------------------------------------+

 =============================== Installation ===============================
  1)  Install AI2Pot                  2)  Install LAMMPS with AI2Pot
 ============================== Preprocessing ===============================
 11)  Convert Dataset                12)  Standardize ExtXYZ
 13)  Analyse Dataset                14)  MTP Active Learning
 15)  NEP Active Learning
 ========================= Potential Training Input =========================
 21)  MTP Training Input             22)  NEP Training Input
 ============================== Postprocessing ==============================
 31)  Plot E/F/V Parity              32)  Plot Learning Curve
 33)  Plot Descriptor Projection     34)  Export TorchScript Model
 =============================== MD Utilities ===============================
 91)  Doctor                         92)  Show Examples
 93)  Print Version
  0)  Quit
 ------------>>
```

Users can select a task by entering the corresponding number.

### 2. Command-line mode

AI2Pot-cli can also be used directly from the command line. For example, a training task can be launched using:

```bash
$ ai2pot-cli train --input xxx_train.jsonc
```

where `xxx_train.jsonc` is the training configuration file generated manually or through the interactive input-generation workflow.

## License

AI2Pot-cli is released under the GNU General Public License v3.0.

See the `LICENSE` file for details.

