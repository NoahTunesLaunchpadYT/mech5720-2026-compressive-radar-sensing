# Coded-aperture simulation

Use `preview` for one ideal scalar diffraction pattern, `run` for an openEMS FDTD solve with propagated fields, and `preview-queue` to evaluate a strip of masks. `preview-super-queue` runs several strip queues in order.

## 1. Set up Windows PowerShell

Open PowerShell in the `Single-pixel-radar` repository root. Use the same Python interpreter for installation and for every command below. This repository has been checked with Python 3.14.

```powershell
python --version
python -m pip install numpy scipy matplotlib h5py
```

Install the Windows build of ParaView. The simulator locates `pvpython.exe` on `PATH`, under `Program Files`, or under `%LOCALAPPDATA%\Programs`. For a custom install, add the folder containing `pvpython.exe` to `PATH`. [ParaView downloads](https://www.paraview.org/Download/)

Verify the preview dependencies. The second command should print the path to `pvpython.exe`.

```powershell
python -c "import numpy, scipy, matplotlib, h5py"
python -c "from coded_aperture_sim.paraview_view import find_pvpython; print(find_pvpython())"
```

For the full-wave `run` and `smoke` commands, extract the Windows openEMS package and install the bundled wheels for your Python version. The example assumes `C:\openEMS` contains the DLLs and `C:\openEMS\python` contains the wheels. Set the DLL path in each new PowerShell session, or set it in your Windows user environment. [openEMS Windows installation](https://docs.openems.de/en/v0.37.0/install/manual_install.html#windows)

```powershell
python -m pip install --no-index --find-links "C:\openEMS\python" openEMS
$env:CSXCAD_INSTALL_PATH = 'C:\openEMS'
python -c "import openEMS, CSXCAD"
```

## 2. Run one scalar preview

1. Save this JSON as `coded_aperture_sim\scene_quick.json`. It overrides the sampling in [scene.json](scene.json) and inherits the default mask and geometry.

   ```json
   {
     "capture_size_mm": [120, 120],
     "capture_step_mm": 1,
     "volume_size_mm": [80, 80, 100],
     "volume_step_mm": 4,
     "preview_depths_mm": [20, 50, 100]
   }
   ```

2. Run the preview from the repository root. Add `--reflections` to include one approximate reflected roundtrip between the mask front and the front plate.

   ```powershell
   python -m coded_aperture_sim preview --config .\coded_aperture_sim\scene_quick.json
   ```

3. Open the printed `coded_aperture_sim\previews\run_*` folder. View `view.png` or use **File > Load State** in ParaView to open `view.pvsm`. Look for `transverse_50.png` and `transverse_50.csv` for the 50 mm intensity slice. The `_normalized` maps scale each slice to its own peak. The `_radar.png` variants reverse x for a view looking along +z with +y up. Each run also saves `scene.json`, `metadata.json`, `volume.h5`, centre cuts, and far-field plots.

Plain `preview` uses [scene_lowres.json](scene_lowres.json). That preset covers an 800 × 800 × 1,250 mm volume at 2 mm spacing and produces slices at 50, 250, 500, and 1,250 mm. Use `--config` to choose a scene and `--output` to choose the parent folder for a fresh timestamped run.

## 3. Run one full-wave simulation

1. Check the solver setup and RAM estimate for the scene you plan to run.

   ```powershell
   python -m coded_aperture_sim check --config .\coded_aperture_sim\scene.json
   ```

   Look for an empty `import_errors` list, a `pvpython` path, and `resource_gate: true`. The estimate must fit both `memory_limit_gib` and available physical RAM after the reserve. The current `scene_lowres.json` uses a coarser mesh but a larger capture plane. Check it separately before using it for FDTD.

2. Run the same scene once preflight passes. The simulator solves the coded mask, solves an all-open reference with the same excitation, then validates the handoff and exports the results.

   ```powershell
   python -m coded_aperture_sim run --config .\coded_aperture_sim\scene.json
   ```

3. Use the timestamped folder under `coded_aperture_sim\runs`, or set a different parent with `--output`. Read `validation.json` for the handoff comparison and the pattern-to-reference forward-flux ratio. Inspect `fields.h5` for the direct solver fields, `volume.h5` for the propagated E/H fields, `reference\` for the all-open run, and `view.pvsm` for the ParaView state. A failed run keeps partial output with `failure.json` and any solver logs.

Run the independent tiny solver diagnostic with a fresh output directory. To rebuild visualizations from saved full-wave fields, pass the actual run directory to `export`.

```powershell
python -m coded_aperture_sim smoke --output .\coded_aperture_sim\smoke-check
python -m coded_aperture_sim export "path\to\run_TIMESTAMP"
```

## 4. Run a batch of strip windows

1. Pick [queue_impulse_mask.json](queue_impulse_mask.json) or [queue_mux_mask.json](queue_mux_mask.json). Both use `scene_lowres.json` and enable reflections. For a short trial, copy one to `queue_quick.json` in the same folder, set `"scene_config": "scene_quick.json"`, and add `"windows": [0, 1]`. The scene path is relative to the queue config.

2. Start the trial or use a supplied config for all 24 windows.

   ```powershell
   python -m coded_aperture_sim preview-queue --config .\coded_aperture_sim\queue_quick.json
   python -m coded_aperture_sim preview-queue --config .\coded_aperture_sim\queue_impulse_mask.json
   ```

   A strip has four 27-bit strings in `strip_rows`, ordered from +y to −y. Window `i` takes columns `i` through `i+3` from each row, starting at zero. Omitting `windows` runs indices 0 through 23 in order. An explicit list keeps its order and repeats. Each window gets its own scalar preview and intensity normalization.

3. Find the printed queue folder under `coded_aperture_sim\preview_queues`, or choose another parent with `--output`. `queue.json` records the resolved configuration, job status, and each output path. Load `queue_view.pvsm` in ParaView and use the animation controls to step through the completed masks. Keep the `.pvd` series files and child run folders together.

4. After Ctrl+C or a failure, pass the existing queue folder to `--resume`. Resume uses its saved configuration, skips completed masks, and retries unfinished jobs in fresh folders. It also retries failed sequence export. `status: completed` records execution. `scientific_status: incomplete` records the preview's validation status.

   ```powershell
   python -m coded_aperture_sim preview-queue --resume ".\coded_aperture_sim\preview_queues\run_TIMESTAMP"
   ```

### Run several queues in order

[super_queue_example.json](super_queue_example.json) lists the impulse and mux queue files in order. Child config paths are relative to the super-queue config. The parent snapshots each child and stops when a child fails.

```powershell
python -m coded_aperture_sim preview-super-queue --config .\coded_aperture_sim\super_queue_example.json
python -m coded_aperture_sim preview-super-queue --resume ".\coded_aperture_sim\preview_super_queues\run_TIMESTAMP"
```

Resume takes the folder containing `super_queue.json`. It skips completed children and resumes unfinished windows in the interrupted child. Each `queue_001`, `queue_002`, and subsequent child folder has its own manifest and ParaView state. Use `--output` only when starting a new queue or super-queue.

Choose a short output path on Windows. Deeply nested run folders can cause ParaView to fail to open its generated export script. If this happens, start a new super-queue with `--output` set to a shorter writable path.

## Scene settings and model limits

- Lengths are in millimetres and frequency is in GHz. Set `mask_rows` to four 4-bit strings. Rows run from +y to −y, columns from −x to +x, and `1` means open. Adjacent openings join. The area outside the 4 × 4 active grid is metal.
- `volume_size_mm` is `[x, y, z]`. The x and y axes centre on the mask axis, and the volume starts at the mask exit. Use dimensions divisible by `volume_step_mm`. The capture plane must cover the opening, active grid, and volume cross-section. `preview_depths_mm` selects depths from the mask exit, within the volume. Slice files use the nearest sampled plane.
- The default PEC enclosure has a 100 × 100 mm front centred at (0, −10) mm and a 26.5 mm square opening centred at (0, 0). The 2 mm walls span z = −3.1 to 41.9 mm. The 2 mm front spans 41.9 to 43.9 mm, and the 1.5 mm back spans −4.6 to −3.1 mm. The ideal x-polarized dipole at the origin radiates in both z directions. The viewing direction is +z.
- The scalar preview propagates an ideal dipole field through two screens and approximates transmission through the mask thickness as free-space propagation. The optional reflection term adds one coherent roundtrip with a coefficient of −1 per bounce. The strip model evaluates discrete mask positions. Its angular patterns are relative scalar intensities. The 200 mm versus 205.2 mm strip discrepancy remains unresolved.
- Full-wave output includes relative `|E|²` and signed `Sz_relative = 0.5 Re((E × H*)z)`. The frequency-domain fields are source-spectrum-weighted, so the flux has arbitrary units and the common-excitation flux ratio is dimensionless. The far-field lobe radius represents normalized amplitude. Full-wave validation remains incomplete pending capture-size convergence and mesh refinement. [Acceptance notes](ACCEPTANCE.md)

## Export saved slices and montages

Export missing transverse CSVs from a saved run or from compatible runs under a parent folder. Each depth produces `transverse_<depth>.csv` and `transverse_<depth>_normalized.csv` with x, y, and intensity columns. Existing CSVs are skipped.

```powershell
python -m coded_aperture_sim.export_transverse_csv "path\to\run_or_parent_folder"
```

Make a 4-column by 6-row montage from a queue containing one completed job per window, with `transverse_1250_normalized.png` in each job folder. The script crops to the map regions and saves `transverse_1250_montage.png` in window-index order.

```powershell
python -m coded_aperture_sim.montage_transverse "path\to\queue_folder"
```
