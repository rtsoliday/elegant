# Agent Notes for elegant

This file contains a short tactical summary based on repository evidence. `../llm-wiki/scripts/refresh_wiki.py` rewrites only the machine-managed block.

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
Add durable repo-specific instructions here.
