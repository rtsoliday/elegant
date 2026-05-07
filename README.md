# elegant

**elegant (ELEctron Generation ANd Tracking)** is a tool for simulating particle accelerators, particularly useful at facilities like the Advanced Photon Source (APS). It helps model beam dynamics, track particles in 6D, and optimize accelerator parameters, making it vital for research and operations.

## Introduction to elegant

**elegant**, standing for ELEctron Generation ANd Tracking, is a 6-D accelerator simulation code developed by the Accelerator Operations and Physics Group at the Advanced Photon Source (APS), a facility under Argonne National Laboratory. It is designed to model and simulate particle accelerators, particularly synchrotron light sources, with capabilities extending to tracking particles in 6-dimensional phase space (x, x', y, y', s, δ), using matrices up to third order, canonical kick elements, and numerically integrated elements. The software is not standalone, relying on the SDDS (Self-Describing Data Sets) Toolkit for post-processing and data analysis, and supports multi-stage simulations, making it suitable for complex projects like start-to-end jitter analysis and top-up tracking.
The program’s philosophy emphasizes flexibility, encouraging users to leverage UNIX shell scripts and languages like Tcl/Tk for tailored outputs. It supports concurrent computing on multiple workstations, with complete backward compatibility to ensure existing input files remain functional. This makes **elegant** an essential tool for the design, operation, and upgrade of particle accelerators, particularly in research environments.

## Documentation

For comprehensive details and examples, please refer to the **[elegant User’s Manual](https://ops.aps.anl.gov/manuals/elegant_latest/elegant.pdf)**.
An [online forum](https://www3.aps.anl.gov/forums/elegant/) is available for support.

## Installation

Clone the repository and build the project:

```bash
git clone https://github.com/rtsoliday/SDDS.git
git clone https://github.com/rtsoliday/elegant.git
cd elegant
make -j
```

The top-level build automatically builds `Pelegant` when MPI compiler wrappers are found.  It also automatically builds the CUDA executables when a usable CUDA Toolkit is found, while keeping the ordinary CPU binaries available.  On a CUDA-capable build host, `make` installs:

```text
bin/Linux-x86_64/elegant
bin/Linux-x86_64/Pelegant
bin/Linux-x86_64-gpu/gpu-elegant
bin/Linux-x86_64-gpu/gpu-Pelegant
```

Use `make CUDA_AUTO=0 -j` for a CPU-only build, or force a specific CUDA compiler with `make HAVE_CUDA=1 NVCC=/usr/local/cuda-12.4/bin/nvcc -j`.  CUDA runtime controls and verification workflows are documented in [doc/CUDA_GPU_SUPPORT.md](doc/CUDA_GPU_SUPPORT.md).

## Usage

Run **elegant** with an appropriate input file:

```bash
./elegant your_input_file.ele
```

Output files are generated in SDDS format and can be post-processed with the SDDS Toolkit.

## Contributing

Contributions, bug reports, and suggestions are welcome. Please open an issue or submit a pull request with your improvements.

## Citation

If you use **elegant** in your research, please cite:

> M. Borland, “elegant: A Flexible SDDS-Compliant Code for Accelerator Simulation,” Advanced Photon Source LS-287, September 2000.

## Authors
- Michael Borland
- Bob Soliday
- Yusong Wang
- Aimin Xiao
- Hairong Shang
- Nikita Kuklev
- Joe Calvey
- Yipeng Sun
- Louis Emery
- Xuesong Jiao
- Vadim Sajaev

## Acknowledgments
This project is developed and maintained by **[Accelerator Operations & Physics](https://www.aps.anl.gov/Accelerator-Operations-Physics)** at the **Advanced Photon Source** at **Argonne National Laboratory**.

For more details, visit the official **[SDDS documentation](https://www.aps.anl.gov/Accelerator-Operations-Physics/Documentation)**.
