# AI2Pot-cli
Command-Line Interface for AI2Pot.

```shell
$ ai2pot-cli 
 +--------------------------------------------------------------------------+
 |                       AI2Pot-cli Standard Edition                        |
 |                           Version 0.1.0                                  |
 |                                                                          |
 |                Official Command Line Interface for AI2Pot                |
 |                                                                          |
 |              Developer: Hanyu Liu (domainofbuaa@gmail.com)               |
 |                                                                          |
 |            AI2Pot-cli : https://github.com/lhycms/AI2Pot-cli             |
 |                AI2Pot : https://github.com/lhycms/AI2Pot                 |
 +--------------------------------------------------------------------------+

 ============================== Preprocessing ===============================
  1)  Convert Dataset                 2)  Standardize ExtXYZ
  3)  Analyse Dataset                 4)  MTP Active Learning
  5)  NEP Active Learning
 ========================= Potential Training Input =========================
 11)  MTP Training Input             12)  NEP Training Input
 ============================== Postprocessing ==============================
 21)  Plot E/F/V Parity              22)  Plot Learning Curve
 =============================== MD Utilities ===============================
 91)  Doctor                         92)  Show Examples
 93)  Print Version
  0)  Quit
 ------------>> 
 
$ ai2pot-cli train --input xxx_train.jsonc
```