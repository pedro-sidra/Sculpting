# Class-wise IoU tables

Generated with `python3 tools/build_classwise_tables.py`.

Class rows are IoU; the final mIoU row is their unweighted mean. Markdown and LaTeX values are percentages rounded to two decimal places. The four CSV tables preserve source precision on the 0–1 scale. Each variant also has a .tex table fragment requiring no extra LaTeX packages.

All 20 runs pass checks against both the report values pinned in the script and `tools/results.csv`, with absolute tolerance 1e-12. Class means also match the stored mIoUs at that tolerance. See `run_manifest.csv` for source paths, metadata, values, and errors.

For MAE+AM+Sculpt Mixed LitePT (artifact key pretrain-sculptor-litept), LR1–LR20 use runs named FT-reconstructor-litept with recorded parent k661gax5 / pretrain-sculptor-litept. LR100 uses run z5qoelcv from the separate FT-reconstructor-litept row in results.csv, as specified by the report author. That run has no parent metadata in experiments_table.csv. Class names retain the source spelling.

## SPUnet from scratch

| Class / metric | LR1 | LR5 | LR10 | LR20 | LR100 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Run ID | `gsntq83r` | `rla9ndef` | `58m7cdcr` | `pqm95tdv` | `qs4z9sr3` |
| wall | 64.28 | 73.41 | 75.93 | 79.49 | 84.74 |
| floor | 91.57 | 94.22 | 94.59 | 94.87 | 95.43 |
| cabinet | 10.49 | 38.91 | 48.48 | 56.19 | 66.72 |
| bed | 27.68 | 56.94 | 64.47 | 73.74 | 81.37 |
| chair | 60.56 | 77.44 | 85.80 | 88.63 | 91.52 |
| sofa | 19.05 | 59.91 | 75.70 | 78.72 | 81.49 |
| table | 36.37 | 53.94 | 59.57 | 65.88 | 73.80 |
| door | 6.92 | 35.75 | 45.48 | 55.03 | 66.73 |
| window | 12.17 | 41.07 | 45.50 | 55.12 | 65.45 |
| bookshelf | 32.74 | 19.97 | 69.83 | 70.44 | 76.26 |
| picture | 0.34 | 8.99 | 19.85 | 29.27 | 31.79 |
| counter | 4.75 | 32.06 | 49.63 | 58.43 | 68.24 |
| desk | 8.10 | 37.79 | 48.10 | 52.72 | 62.82 |
| curtain | 42.42 | 50.02 | 61.25 | 45.89 | 61.30 |
| refridgerator | 2.30 | 18.26 | 31.13 | 33.65 | 55.72 |
| shower curtain | 0.00 | 8.98 | 58.81 | 69.62 | 71.33 |
| toilet | 0.00 | 68.07 | 82.67 | 89.70 | 94.88 |
| sink | 0.00 | 31.82 | 49.22 | 64.77 | 68.51 |
| bathtub | 0.00 | 52.72 | 84.08 | 83.53 | 87.38 |
| otherfurniture | 1.56 | 23.06 | 28.05 | 33.83 | 59.37 |
| mIoU | 21.07 | 44.17 | 58.91 | 63.98 | 72.24 |

## MAE+AM+Sculpt Mixed SPUnet

| Class / metric | LR1 | LR5 | LR10 | LR20 | LR100 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Run ID | `kbjmpqnq` | `ivbcujrw` | `vgufxrmd` | `bqgxvp4u` | `ke3or8ft` |
| wall | 67.47 | 77.81 | 80.05 | 82.51 | 85.99 |
| floor | 92.15 | 94.75 | 95.22 | 95.30 | 95.84 |
| cabinet | 25.13 | 41.00 | 48.56 | 55.17 | 66.98 |
| bed | 36.36 | 63.37 | 67.39 | 68.61 | 83.62 |
| chair | 61.97 | 77.55 | 84.84 | 88.48 | 91.94 |
| sofa | 32.61 | 57.05 | 72.15 | 75.90 | 84.33 |
| table | 42.33 | 58.68 | 62.22 | 66.97 | 76.29 |
| door | 16.00 | 44.29 | 52.32 | 56.96 | 68.25 |
| window | 20.49 | 49.17 | 52.45 | 59.96 | 67.51 |
| bookshelf | 39.35 | 31.31 | 70.40 | 72.88 | 82.07 |
| picture | 6.21 | 25.55 | 24.64 | 32.54 | 37.08 |
| counter | 23.54 | 43.37 | 57.03 | 63.28 | 70.38 |
| desk | 16.19 | 45.49 | 51.39 | 56.33 | 68.27 |
| curtain | 47.44 | 60.28 | 66.68 | 65.12 | 77.25 |
| refridgerator | 7.05 | 24.91 | 31.13 | 38.42 | 60.21 |
| shower curtain | 0.00 | 20.02 | 59.71 | 55.77 | 66.55 |
| toilet | 0.00 | 70.97 | 84.91 | 89.19 | 93.27 |
| sink | 0.00 | 34.11 | 50.85 | 62.64 | 66.60 |
| bathtub | 0.00 | 72.06 | 85.38 | 87.04 | 89.72 |
| otherfurniture | 10.67 | 25.29 | 29.98 | 33.94 | 60.25 |
| mIoU | 27.25 | 50.85 | 61.37 | 65.35 | 74.62 |

## LitePT from scratch

| Class / metric | LR1 | LR5 | LR10 | LR20 | LR100 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Run ID | `kjat7htt` | `iqe1zpo4` | `6sy5ykoy` | `hbz5jpv6` | `2849ch9g` |
| wall | 65.99 | 73.30 | 77.32 | 80.95 | 85.80 |
| floor | 89.71 | 94.36 | 94.57 | 94.98 | 95.32 |
| cabinet | 20.37 | 39.47 | 50.67 | 53.80 | 72.40 |
| bed | 39.60 | 68.80 | 75.00 | 77.44 | 81.78 |
| chair | 54.87 | 75.76 | 85.72 | 87.13 | 90.21 |
| sofa | 31.58 | 55.79 | 72.49 | 76.39 | 81.63 |
| table | 37.94 | 57.93 | 57.22 | 65.02 | 78.07 |
| door | 19.84 | 39.47 | 48.61 | 57.36 | 68.39 |
| window | 14.55 | 45.32 | 51.83 | 60.00 | 68.35 |
| bookshelf | 27.25 | 12.91 | 73.30 | 71.06 | 82.35 |
| picture | 12.53 | 17.69 | 25.07 | 32.93 | 34.13 |
| counter | 22.38 | 38.41 | 59.86 | 55.64 | 66.15 |
| desk | 14.23 | 42.15 | 47.16 | 53.29 | 73.36 |
| curtain | 36.23 | 56.15 | 66.38 | 58.22 | 78.94 |
| refridgerator | 5.15 | 18.73 | 33.90 | 44.98 | 67.40 |
| shower curtain | 0.00 | 26.87 | 67.49 | 64.85 | 72.05 |
| toilet | 0.00 | 68.35 | 84.16 | 91.68 | 94.79 |
| sink | 12.72 | 32.95 | 45.08 | 64.03 | 69.71 |
| bathtub | 0.00 | 64.94 | 84.22 | 82.78 | 86.14 |
| otherfurniture | 9.09 | 21.83 | 27.96 | 32.59 | 59.61 |
| mIoU | 25.70 | 47.56 | 61.40 | 65.26 | 75.33 |

## MAE+AM+Sculpt Mixed LitePT

| Class / metric | LR1 | LR5 | LR10 | LR20 | LR100 |
| :--- | ---: | ---: | ---: | ---: | ---: |
| Run ID | `ea7k5e6a` | `mas2tjhk` | `7uk60p4e` | `6tnedmix` | `z5qoelcv` |
| wall | 67.85 | 77.34 | 80.22 | 83.06 | 86.45 |
| floor | 92.26 | 94.50 | 95.33 | 95.33 | 95.59 |
| cabinet | 23.03 | 43.53 | 52.61 | 57.90 | 69.58 |
| bed | 40.11 | 63.46 | 73.87 | 79.16 | 81.03 |
| chair | 57.03 | 75.77 | 84.05 | 85.48 | 90.64 |
| sofa | 24.82 | 51.86 | 65.18 | 69.77 | 80.58 |
| table | 41.91 | 59.91 | 62.78 | 69.79 | 78.61 |
| door | 16.65 | 43.17 | 54.51 | 61.14 | 70.96 |
| window | 14.73 | 52.20 | 53.79 | 63.10 | 71.52 |
| bookshelf | 32.76 | 22.60 | 75.98 | 79.60 | 84.07 |
| picture | 13.98 | 25.24 | 29.31 | 31.96 | 35.08 |
| counter | 29.26 | 50.34 | 57.52 | 58.23 | 67.56 |
| desk | 8.50 | 43.19 | 54.60 | 57.75 | 72.44 |
| curtain | 42.85 | 57.02 | 70.87 | 70.81 | 76.63 |
| refridgerator | 15.26 | 33.17 | 39.72 | 52.66 | 75.88 |
| shower curtain | 0.00 | 25.16 | 68.37 | 65.20 | 72.59 |
| toilet | 0.00 | 66.41 | 82.26 | 93.77 | 93.58 |
| sink | 16.75 | 34.21 | 53.27 | 64.67 | 69.98 |
| bathtub | 0.00 | 73.15 | 86.40 | 87.32 | 88.98 |
| otherfurniture | 9.63 | 23.01 | 30.07 | 38.30 | 56.05 |
| mIoU | 27.37 | 50.76 | 63.54 | 68.25 | 75.89 |
