"""ParaView export helpers (ParaView's own Python runtime owns VTK)."""

import os
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

_VIEW_SETUP = r'''layout=CreateLayout('Coded aperture field and far-field lobes'); layout.SplitHorizontal(0,0.5); view=GetActiveViewOrCreate('RenderView'); AssignViewToLayout(view,layout,0)
disp=Show(volume,view); disp.Representation='Volume'; ColorBy(disp,('POINTS',active_scalar)); disp.SetScalarBarVisibility(view,True)
bar=GetScalarBar(disp.LookupTable,view); bar.Title='Relative intensity'; bar.ComponentTitle=''; bar.AutomaticLabelFormat=1; bar.UseCustomLabels=1; bar.CustomLabels=[0.0,0.333,0.667,1.0]; bar.AddRangeLabels=0; bar.Orientation='Horizontal'; bar.WindowLocation='Any Location'; bar.Position=[0.18,0.12]; bar.ScalarBarLength=0.62; bar.ScalarBarThickness=14; bar.TitleFontSize=12; bar.LabelFontSize=10
slice_filter=Slice(Input=volume); slice_filter.SliceType='Plane'; slice_filter.SliceType.Origin=center; slice_filter.SliceType.Normal=(0,0,1)
slice_disp=Show(slice_filter,view); ColorBy(slice_disp,('POINTS',active_scalar)); slice_disp.SetScalarBarVisibility(view,True)
bar=GetScalarBar(slice_disp.LookupTable,view); bar.Title='Relative intensity'; bar.ComponentTitle=''; bar.AutomaticLabelFormat=1; bar.UseCustomLabels=1; bar.CustomLabels=[0.0,0.333,0.667,1.0]; bar.AddRangeLabels=0; bar.Orientation='Horizontal'; bar.WindowLocation='Any Location'; bar.Position=[0.18,0.12]; bar.ScalarBarLength=0.62; bar.ScalarBarThickness=14; bar.TitleFontSize=12; bar.LabelFontSize=10
gdisp=Show(geometry,view); gdisp.Opacity=0.4
status=scene.get('validation_status','not recorded'); shown_status='HANDOFF PASSED' if status=='passed' else 'UNVALIDATED'
title=Text(Text='IDEAL SCALAR DIFFRACTION — NOT FULL-WAVE VALIDATED' if preview else 'Field volume | '+shown_status); title_disp=Show(title,view); title_disp.FontSize=14
if sequence_mode:
    frame_label=Text(Text='Queue frame index is the ParaView time value'); frame_disp=Show(frame_label,view); frame_disp.FontSize=11
view.CameraFocalPoint=center; view.CameraPosition=(center[0]+1.5*scale,center[1]-2*scale,center[2]+1.5*scale); view.CameraViewUp=(0,0,1); view.ResetCamera()
far=CreateView('RenderView'); AssignViewToLayout(far,layout,1); far.OrientationAxesVisibility=1
far_disp=Show(farfield,far); ColorBy(far_disp,('POINTS','normalized_radius')); far_disp.SetScalarBarVisibility(far,True)
farbar=GetScalarBar(far_disp.LookupTable,far); farbar.Title='Relative scalar amplitude' if preview else 'Relative field amplitude'; farbar.ComponentTitle=''; farbar.AutomaticLabelFormat=1; farbar.UseCustomLabels=1; farbar.CustomLabels=[0.0,0.333,0.667,1.0]; farbar.AddRangeLabels=0; farbar.Orientation='Horizontal'; farbar.WindowLocation='Any Location'; farbar.Position=[0.18,0.12]; farbar.ScalarBarLength=0.62; farbar.ScalarBarThickness=14; farbar.TitleFontSize=12; farbar.LabelFontSize=10
lobetitle=Text(Text='Scalar angular pattern | normalized radius' if preview else 'Far-field lobe | unitless radius'); lobe_title_disp=Show(lobetitle,far); lobe_title_disp.FontSize=14
far.CameraFocalPoint=(0,0,0); far.CameraPosition=(2,-2,2); far.CameraViewUp=(0,0,1); far.ResetCamera()
SetActiveView(view); layout.SetSize(1600,900)
if sequence_mode: SaveState(os.path.join(root,'queue_view.pvsm'))
else: SaveScreenshot(os.path.join(root,'view.png'),layout,ImageResolution=[1600,900],SaveAllViews=1); SaveState(os.path.join(root,'view.pvsm'))
'''


def find_pvpython():
    found = shutil.which("pvpython") or shutil.which("pvpython.exe")
    if found:
        return Path(found)
    roots = [os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")]
    for root in filter(None, roots):
        base = Path(root)
        candidates = list(base.glob("ParaView*/bin/pvpython.exe"))
        candidates += list(base.glob("Programs/ParaView*/bin/pvpython.exe"))
        if candidates:
            return candidates[-1]
    return None


def write_paraview_files(run_dir, scene_path, volume_raw, dims, origin, spacing):
    """Write binary VTK datasets and a saved ParaView state in pvpython."""
    import json
    import subprocess
    import tempfile

    pvpython = find_pvpython()
    if pvpython is None:
        raise RuntimeError("ParaView pvpython executable was not found")
    run_dir = Path(run_dir).resolve()
    script = r'''import json, os, sys
import numpy as np
from paraview.simple import *
from paraview import servermanager
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkMultiBlockDataSet, vtkPolyData, vtkCellArray, vtkCompositeDataSet
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkFiltersSources import vtkCubeSource, vtkPlaneSource
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter, vtkXMLMultiBlockDataWriter, vtkXMLPolyDataWriter
root, scene_path, raw_path, nx, ny, nz, ox, oy, oz, dx, dy, dz = sys.argv[1:]
nx, ny, nz = map(int, (nx, ny, nz)); ox, oy, oz, dx, dy, dz = map(float, (ox, oy, oz, dx, dy, dz))
scene = json.load(open(scene_path, encoding='utf-8')); preview = scene.get('preview_model') is True
image = vtkImageData(); image.SetDimensions(nx, ny, nz); image.SetOrigin(ox, oy, oz); image.SetSpacing(dx, dy, dz)
scalars = [('relative_intensity', 'relative_intensity.f32')] if preview else [('relative_E2', 'relative_E2.f32'), ('Sz_relative', 'Sz_relative.f32')]
for name, filename in scalars:
    a = np.fromfile(os.path.join(raw_path, filename), dtype=np.float32)
    if a.size != nx*ny*nz: raise RuntimeError('bad volume scalar size: '+name)
    vtk = numpy_to_vtk(a, deep=True); vtk.SetName(name); image.GetPointData().AddArray(vtk)
active_scalar = 'relative_intensity' if preview else 'relative_E2'
image.GetPointData().SetActiveScalars(active_scalar)
w = vtkXMLImageDataWriter(); w.SetFileName(os.path.join(root, 'volume.vti')); w.SetInputData(image); w.SetDataModeToAppended(); w.SetCompressorTypeToZLib(); assert w.Write()
blocks = vtkMultiBlockDataSet(); boxes = scene.get('geometry')
if boxes is None:
    from coded_aperture_sim.model import geometry
    boxes = geometry(scene)
blocks.SetNumberOfBlocks(len(boxes))
for i, b in enumerate(boxes):
    if b['start'][2] == b['stop'][2]:
        plane = vtkPlaneSource(); plane.SetOrigin(b['start'][0], b['start'][1], b['start'][2]); plane.SetPoint1(b['stop'][0], b['start'][1], b['start'][2]); plane.SetPoint2(b['start'][0], b['stop'][1], b['start'][2]); plane.Update(); blocks.SetBlock(i, plane.GetOutput())
    else:
        cube = vtkCubeSource(); cube.SetBounds(*(b['start'][0], b['stop'][0], b['start'][1], b['stop'][1], b['start'][2], b['stop'][2])); cube.Update(); blocks.SetBlock(i, cube.GetOutput())
    blocks.GetMetaData(i).Set(vtkCompositeDataSet.NAME(), b['name'])
w = vtkXMLMultiBlockDataWriter(); w.SetFileName(os.path.join(root, 'geometry.vtm')); w.SetInputData(blocks); w.SetDataModeToAppended(); assert w.Write()
ff = np.load(os.path.join(root, 'farfield.npz')); theta, phi, power = ff['theta_rad'], ff['phi_rad'], ff['power']
th, ph = np.meshgrid(theta, phi, indexing='ij')
if not np.all(np.isfinite(power)) or np.any(power < 0): raise ValueError('far-field power must be finite and nonnegative')
max_power = float(np.max(power)); radius = np.sqrt(power/max_power) if max_power > 0 else np.zeros_like(power)
pts = vtkPoints(); polys = vtkCellArray(); ids = np.empty((len(theta), len(phi)), dtype=np.int64)
for j in range(len(theta)):
    for i in range(len(phi)):
        ids[j,i] = pts.InsertNextPoint(radius[j,i]*np.sin(th[j,i])*np.cos(ph[j,i]), radius[j,i]*np.sin(th[j,i])*np.sin(ph[j,i]), radius[j,i]*np.cos(th[j,i]))
for j in range(len(theta)-1):
    for i in range(len(phi)):
        ni=(i+1)%len(phi); cell=[ids[j,i],ids[j,ni],ids[j+1,ni],ids[j+1,i]]; polys.InsertNextCell(4)
        for k in cell: polys.InsertCellPoint(int(k))
poly=vtkPolyData(); poly.SetPoints(pts); poly.SetPolys(polys); arr=numpy_to_vtk(radius.ravel(),deep=True); arr.SetName('normalized_radius'); poly.GetPointData().AddArray(arr)
w=vtkXMLPolyDataWriter(); w.SetFileName(os.path.join(root,'farfield.vtp')); w.SetInputData(poly); w.SetDataModeToAppended(); assert w.Write()
Disconnect(); Connect()
volume=XMLImageDataReader(FileName=[os.path.join(root,'volume.vti')]); geometry=XMLMultiBlockDataReader(FileName=[os.path.join(root,'geometry.vtm')]); farfield=XMLPolyDataReader(FileName=[os.path.join(root,'farfield.vtp')])
center=(ox+dx*(nx-1)/2,oy+dy*(ny-1)/2,oz+dz*(nz-1)/2); scale=max(nx*dx,ny*dy,nz*dz); sequence_mode=False
__VIEW_SETUP__
'''
    script = script.replace("__VIEW_SETUP__", _VIEW_SETUP)
    with tempfile.TemporaryDirectory(prefix=".coded_aperture_pv_", dir=run_dir) as tmp:
        script_path = Path(tmp) / "export.py"
        script_path.write_text(script, encoding="utf-8")
        args = [str(pvpython), str(script_path), str(run_dir), str(scene_path), str(volume_raw),
                *map(str, dims), *map(str, origin), *map(str, spacing)]
        subprocess.run(args, check=True, cwd=run_dir, timeout=120)


def write_paraview_sequence(queue_dir, completed_runs):
    """Write queue-wide PVD collections and a ParaView state for completed runs."""
    import subprocess
    import tempfile

    completed_runs = sorted(completed_runs, key=lambda item: item[0])
    if not completed_runs:
        return
    queue_dir = Path(queue_dir).resolve()
    queue_dir.mkdir(parents=True, exist_ok=True)
    pvpython = find_pvpython()
    if pvpython is None:
        raise RuntimeError("ParaView pvpython executable was not found")

    for name, dataset in (("volume", "volume.vti"), ("geometry", "geometry.vtm"), ("farfield", "farfield.vtp")):
        collection = ET.Element("VTKFile", type="Collection", version="0.1", byte_order="LittleEndian")
        frames = ET.SubElement(collection, "Collection")
        for position, run_dir in completed_runs:
            run_dir = Path(run_dir).resolve()
            ET.SubElement(frames, "DataSet", timestep=str(position), group="", part="0",
                          file=os.path.relpath(run_dir / dataset, queue_dir).replace(os.sep, "/"))
        payload = ET.tostring(collection, encoding="utf-8", xml_declaration=True)
        fd, temp_path = tempfile.mkstemp(prefix=f".{name}.", suffix=".pvd", dir=queue_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
            os.replace(temp_path, queue_dir / f"{name}.pvd")
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

    script = r'''import os, sys
from paraview.simple import *
root=sys.argv[1]
Disconnect(); Connect()
volume=PVDReader(FileName=os.path.join(root,'volume.pvd'))
geometry=PVDReader(FileName=os.path.join(root,'geometry.pvd'))
farfield=PVDReader(FileName=os.path.join(root,'farfield.pvd'))
volume.UpdatePipeline(); bounds=volume.GetDataInformation().GetBounds(); center=((bounds[0]+bounds[1])/2,(bounds[2]+bounds[3])/2,(bounds[4]+bounds[5])/2); scale=max(bounds[1]-bounds[0],bounds[3]-bounds[2],bounds[5]-bounds[4])
preview=True; active_scalar='relative_intensity'; scene={}; sequence_mode=True
scene_proxy=GetAnimationScene(); scene_proxy.UpdateAnimationUsingDataTimeSteps()
__VIEW_SETUP__
'''
    script = script.replace("__VIEW_SETUP__", _VIEW_SETUP)
    with tempfile.TemporaryDirectory(prefix=".coded_aperture_sequence_", dir=queue_dir) as tmp:
        script_path = Path(tmp) / "sequence.py"
        script_path.write_text(script, encoding="utf-8")
        subprocess.run([str(pvpython), str(script_path), str(queue_dir)], check=True, cwd=queue_dir, timeout=120)
