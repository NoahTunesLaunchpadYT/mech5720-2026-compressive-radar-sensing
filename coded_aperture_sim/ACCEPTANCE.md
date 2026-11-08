# Bounded implementation acceptance — 29 October 2026

The orchestrator reviewed the implementation and returned blocking issues to the three Luna implementation agents. The final code review accepts the bounded-check scope. This is not acceptance of physical prediction accuracy.

## Verified

- All 15 automated tests passed under the existing Python 3.14 installation, including the actual ParaView export fixture.
- Actual openEMS smoke diagnostics produced finite complex fields and native near-to-far-field data.
- The non-square diagnostic preserved uniform x/y coordinates and the expected x-polarized dipole orientation; its measured x-offset/y-offset intensity ratio was about 0.439, with positive integrated forward flux.
- The reduced production-path diagnostic in `small_end_to_end` reached the requested -50 dB energy criterion after 1,344 timesteps. This used reduced test geometry, not the user's full geometry.
- The orchestrator visually inspected `outputs/actual_export_qa/view.png`: the field volume, movable slice, translucent geometry, angular lobes, legends and unvalidated status are visible.
- A mocked low-memory check proved that the CLI rejects the run before creating an output directory or invoking the solver.
- Existing repository files are unchanged; implementation and diagnostic artifacts are confined to this folder. No virtual environment or new packages were installed.

## Scientific limits and pending checks

- The reduced diagnostic's propagation comparison FAILED the 5% threshold: direct-plane electric-field relative errors were approximately 21%, 43% and 73%. Its small capture also failed padding convergence. The diagnostic is explicitly marked UNVALIDATED, and the CLI rejects failed handoff validation.
- Full-size simulation, capture-size convergence and mesh-refinement convergence were not run, as requested. The diagnostic failure must not be reported as validation of the full scene or hybrid propagation accuracy.
- The default resource estimate is about 19.86 GiB, with a 24 GiB configured limit and an additional live available-RAM gate. The observed available RAM during final checks was about 14–15 GiB; the full run was blocked without starting it.
- Field and flux values retain arbitrary source-spectrum-weighted units. No calibrated watts, absolute gain or actual XM125 beam claim is made.
- ParaView emits MathText initialization warnings while rendering on this installation. The saved VTK data, state and PNG were successfully produced and visually checked.

## Usage

From the Single-pixel-radar project root, run `py -3.14 -m coded_aperture_sim check`. See README.md for run, export, smoke and mask-editing instructions.
