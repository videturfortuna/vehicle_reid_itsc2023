# vehicle_reid_itsc2023
Strength in Diversity: Multi-Branch Representation Learning for
Vehicle Re-Identification

Code will be available upon paper acceptance to ITSC 2023.

When submitted we did not had acess to VehicleID so we report now values for VehicleID:

VehicleID - Half Precision - Baseline mAP:  88.37 CMC1: 82.77
Resuls are displayed as mAP / CMC1 in percentage values %.
ResNet50 only
| R50      | 4G          | 2G          | 2X2G        | 2B          | 4B          |
| -------- | ----------- | ----------- | ----------- | ----------- | ----------- |
| CE + Tri | 89.61 / 84.34 | 88.82 / 83.16 | 89.00 / 83.45 | 88.58 / 82.95 | 88.78 / 83.19 |
| CE or Tri |  /            | 89.96 / 84.88 | 90.5 / 85.35  | 90.6 / 85.83  | 91.0 / 86.19  |
| Params   | 12M         | 16M         | 23.5M       | 38.5M       | 69.6M       |

Hybrid R50+BoT
| R50+BoT  | 4G          | 2G    | 2X2G  | 2B    | 4B          |
| -------- | ----------- | ----- | ----- | ----- | ----------- |
| CE + Tri | 88.12 / 82.23 |  /      |  /      |  /      |  /            |
| CE or Tri | 89.85 / 84.64 | NULL  |  /      | NULL  | 90.99 / 86.08 |
| Params   | 11.7M       | 13.7M | 18.8M | 33.8M | 59.1M       |
