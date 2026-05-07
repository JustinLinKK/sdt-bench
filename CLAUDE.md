# Requirements

- Please find a way to fulfill my RTX 5070Ti so that the best performance can be measured

- Please include **all** the following indicators to represent GPU utility:

| Indicator | Full Name | Meaning |
|---|---|---|
| SM_ACTIVE | Streaming Multiprocessor Active | Fraction of time at least one warp is assigned to an SM (compute occupancy proxy) |
| SM_OCCUPANCY | Streaming Multiprocessor Occupancy | Average ratio of resident warps to max supported warps per SM |
| DRAM_ACTIVE | DRAM Active | Fraction of cycles the device memory interface is busy (HBM/GDDR bandwidth pressure) |
| PCIE_RX | PCIe Receive Bandwidth | Bytes per second received over PCIe (host→device traffic, e.g. data loading) |

# Results

> Save results in `./results/` folder

## Images

- GPU utility - Time

- Agent status(writing code/debug or train model on GPU) - Time

# Others

- Data is available in `/home/downeyflyfan/Research_Projects/AI/Datasets/mle-bench-data`

# TODO

> Try all different kinds of setting:

- upperlimit is dynamic/fixed

- mps/exclusive

- `torch.cuda.stream`/NO Stream

- `torch.multiprocess` /NO multiprocess

- Different NNs (Can combine MLP and CNN / Combine MLP and MLP)
