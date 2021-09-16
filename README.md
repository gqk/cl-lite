## Introduction

The `cl-lite` is a light-weight Class-Incremental Learning (Continual Learning) framework and toolbox based on [PyTorch-Lightning](https://lightning-bolts.readthedocs.io). It basically follows the design of PyTorch-Lightning, but adds support for Class-Incremental Learning, and borrows some useful tools from other [libraries](#acknowledgement).

**Note!!!** It is still in development and will continue to be improved.

## Installation

The simplest way to install `cl-lite` is throght pip as the following command:

```bash
pip install git+https://github.com/gqk/cl-lite.git
```

For more details about installation and requirements, please refer to [Installation](./INSTALLATION.md).

## How To Use

We implement several popular CIL approaches in [cl-lite-projects](https://github.com/gqk/cl-lite-projects), which works with the lastest version of this library. You can quickly start your own project by copying one of these projects, then modiy it to implement your approach.

We do not plan to provide documents for now, please learn from [cl-lite-projects](https://github.com/gqk/cl-lite-projects) and the source code.

## Acknowledgement

Some codes of this library are borrowed from the following open source projects:

- [PODNet Code](https://github.com/arthurdouillard/incremental_learning.pytorch)
- [UCIR Code](https://github.com/hshustc/CVPR19_Incremental_Learning)
- [iCaRL Code](https://github.com/srebuffi/iCaRL)
