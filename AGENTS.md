# Agent Notes for elegant

This file contains a short tactical summary based on repository evidence. `../llm-wiki/scripts/refresh_wiki.py` rewrites only the machine-managed block.

## Operational Security for Automated Tests and Diagnostics

Development and testing occur on monitored laboratory systems. Commands that
combine temporary executable content, runtime injection, wrapper scripts, or
headless process automation may resemble malicious activity and trigger
security alerts even when used for legitimate debugging.

### Default requirements

- Prefer existing repository build and test targets over ad hoc shell workflows.
- Run commands as separate, directly attributable steps. Do not combine
  compilation, permission changes, execution, log collection, and cleanup into
  one `bash -lc` command.
- Place generated executables and shared libraries in the repository's
  documented build or test-artifact directory, not in `/tmp`.
- Do not create executable wrapper scripts dynamically. Prefer checked-in test
  helpers, direct argument-list process execution, or purpose-built test
  binaries.
- Do not apply `chmod +x` to generated content unless the user has explicitly
  approved it. Checked-in scripts should carry their executable bit in Git.
- Prefer supported headless facilities, such as `QT_QPA_PLATFORM=offscreen`,
  over additional display or process wrappers when they satisfy the test.
- Preserve commands, source paths, output paths, and test names in test logs so
  activity can be readily attributed to this repository.

### Security-sensitive techniques

Codex must not introduce or execute any of the following without explicit user
approval in the current conversation:

- `LD_PRELOAD`, `DYLD_INSERT_LIBRARIES`, or other runtime library injection
- `ptrace`, debugger attachment, syscall/API hooking, or symbol interposition
- generated executable content in `/tmp`, `/var/tmp`, or another shared
  temporary directory
- dynamically generated wrapper scripts that execute or replace another program
- privilege changes, capability changes, setuid/setgid behavior, or namespace
  manipulation
- disabling, bypassing, or testing endpoint security controls

Before requesting approval, Codex must explain:

1. why the technique is necessary;
2. which process and files it affects;
3. where generated artifacts will be stored;
4. what safer alternatives were considered;
5. the exact command or repository test target that will run; and
6. how artifacts will be retained or removed.

Approval for one command or test does not authorize unrelated uses of the same
technique.

### Design and test guidance

When a regression test appears to require runtime injection or executable
wrappers, first prefer one of these approaches:

1. Extract the relevant behavior into a directly testable function or component.
2. Add a narrow, documented test interface to the application.
3. Use a checked-in helper program built through the normal build system.
4. Run the security-sensitive integration test only through a named, opt-in
   target on an approved test host or CI runner.

Security-sensitive tests must be clearly named and excluded from routine local
test execution by default. Their documentation should state why the technique
is used and identify the expected processes and artifacts.

If a requested diagnostic cannot be completed without a security-sensitive
technique, stop and ask the user rather than constructing an ad hoc workaround.

<!-- BEGIN MACHINE:summary -->
## Quick start
- Repository-local guidance is sufficient: start with `AGENTS.md`, `README.md`, `docs/`, build/test/config files, and the source tree.
- **elegant (ELEctron Generation ANd Tracking)** is a tool for simulating particle accelerators, particularly useful at facilities like the Advanced Photon Source (APS). It helps model beam dynamics, track particles in 6D, and optimize accelerator parameters, making it vital for research and operations.
- Primary work areas: `doc`, `elegantTools`, `physics`, `ringAnalysisTemplates`, `rpm`, `sddsbrightness`.

## Read first
- `README.md`: Primary project overview and workflow notes
- `doc/Makefile`: Build system entry point or dependency manifest
- `elegantTools/Makefile`: Build system entry point or dependency manifest
- `physics/Makefile`: Build system entry point or dependency manifest
- `sddsbrightness/Makefile`: Build system entry point or dependency manifest

## Build and test
- Documented setup/build commands: `make -j`.
- Detected build systems: GNU Make.
- Unknown: no test workflow evidence was found in the inspected files.
- Likely run commands or operator entry points: `./elegant your_input_file.ele`.

## Operational warnings
- Local checkout layout appears significant; avoid casual changes to sibling-repo assumptions or relative paths.
- Platform-specific dependency setup matters; do not assume one platform's build recipe carries over unchanged.

## Compatibility constraints
- Cross-platform support exists, but platform-specific dependency setup matters.
- Build and runtime behavior likely depends on neighboring core toolkit checkouts.

## Related knowledge
- Repository-local documentation should be treated as authoritative.
- If a shared `llm-wiki/` directory is present in this workspace or parent folder, consult [the matching repo page](../llm-wiki/repos/elegant.md) for additional architectural context.
- If no shared wiki is present, continue using repository-local evidence only.
- If available, [the SDDS concept page](../llm-wiki/concepts/sdds.md) adds broader cross-repo context.
- If present in this workspace, [the cross-repo map](../llm-wiki/insights/cross-repo-map.md) helps explain related repositories.
<!-- END MACHINE:summary -->

## Human notes

### GPU parity requirement
- elegant has CPU and CUDA/GPU implementations that must remain behaviorally in sync.
- Before completing any change that affects tracking, beam state, element behavior, particle loss, RF, wakefields, CSR, LSC, space charge, matrix/magnet/collimator behavior, build flags, or element metadata, check whether the GPU implementation also needs updates.
- Review these areas as applicable:
  - `src/gpu/`, especially `gpu_cuda_runtime.cu`, `gpu_stub.c`, and `gpu_*.h`
  - `#ifdef HAVE_GPU` and `#ifdef GPU_VERIFY` call sites in changed CPU files
  - `src/track_data.c` element flags, especially `GPU_SUPPORT`
  - CPU/GPU particle transfer and synchronization assumptions
  - CUDA-disabled stub behavior so non-CUDA builds still compile
- Do not finish a task until the final response includes one of:
  - `GPU parity: not applicable` with the concrete reason
  - `GPU parity: checked` with the GPU files or call sites reviewed
  - `GPU parity: updated` with the CPU and GPU files changed
- If a CPU behavior change affects a GPU-supported element, update the GPU path in the same change or explicitly call out the required GPU follow-up and why it was not completed.
