# AI2Pot-cli
Command-Line Interface for AI2Pot.

```shell
$ ai2pot-cli   
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
 
$ ai2pot-cli train --input xxx_train.jsonc
```