# vehicle_reid_itsc2023
Strength in Diversity: Multi-Branch Representation Learning for
Vehicle Re-Identification
Version 0.1 I still need to complete somethings but I expect that in October to be finished.
Code will be available upon paper acceptance to ITSC 2023.

All models are trained with CUDA 11.3 and PyTorch 1.11 on RTX4090 and Ryzen 7950X.

Resuls are displayed as mAP / CMC1 in percentage values %.
VehicleID was not available at the time, so we report values for VehicleID (ongoing) now, we follow evaluation as FastReid with 10-fold cross-validation to select queries and gallery.


VehicleID - No Lamba tuning for these results - Half Precision - Baseline mAP:  88.37 CMC1: 82.77


ResNet50 
| R50      | 4G          | 2G          | 2X2G        | 2B          | 4B          |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CE + Tri | 89.61 / 84.34 | 88.82 / 83.16 | 89.00 / 83.45 | 88.58 / 82.95 | 88.78 / 83.19 |
| CE or Tri |  90.31 / 85.27            | 89.96 / 84.88 | 90.5 / 85.35  | 90.6 / 85.83  | 91.0 / 86.19  |
| Params   | 12M         | 16M         | 23.5M       | 38.5M       | 69.6M       |

Hybrid R50+BoT
| R50+BoT  | 4G          | 2G    | 2X2G  | 2B    | 4B          |
| -------- | ----------- | ----- | ----- | ----- | ----------- |
| CE + Tri | 88.12 / 82.23 |  87.68 / 81.73      |  89.02 / 83.58      |  88.87 / 83.28      |  90.06 / 84.78            |
| CE or Tri | 89.85 / 84.64 | NULL  |  90.11 / 84.94     | NULL  | 90.99 / 86.08 |
| Params   | 11.7M       | 13.7M | 18.8M | 33.8M | 59.1M       |



### Veri-776
Full Precision - Baseline mAP: 81.15 CMC1: 96.96 lambda:0.6 beta:1.0

| R50      | 4G          | 2G          | 2X2G        | 2X          | 4X          |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CE + Tri | 82.81/97.38 | 83.04/97.14 | 83.67/97.32 | 81.82/96.96 | 82.31/97.32 |
| CE / Tri | 82.47/96.84 | 83.26/97.02 | 84.22/97.02 | 83.67/97.50 | 83.89/97.5  |

| R50+BoT  | 4G          | 2G           | 2X2G        | 2X          | 4X          |
| -------- | ----------- | ------------ | ----------- | ----------- | ----------- |
| CE + Tri | 82.04/96.96 | 81.14/ 97.02 | 82.02/96.78 | 82.82/97.20 | 83.3/97.62  |
| CE / Tri | 82.67/97.02 | NULL         | 82.57/97.32 | NULL        | **84.72**/**97.68** |
