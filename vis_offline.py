import os
os.environ['QT_PLUGIN_PATH'] = '/usr/lib/x86_64-linux-gnu/qt5'
offscreen = False
if os.environ.get('DISP', 'f') == 'f':
    try:
        from pyvirtualdisplay import Display
        display = Display(visible=False, size=(2560, 1440))
        display.start()
        offscreen = True
    except:
        print("Failed to start virtual display.")

try:
    from mayavi import mlab
    import mayavi
    mlab.options.offscreen = offscreen
    print("Set mlab.options.offscreen={}".format(mlab.options.offscreen))
except:
    print("No Mayavi installation found.")

import torch, numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.style as mplstyle
mplstyle.use('fast')
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib.colors as colors
from pyquaternion import Quaternion
from mpl_toolkits.axes_grid1 import ImageGrid

from model.utils.safe_ops import safe_sigmoid
from tvtk.api import tvtk

# Ensure get_kitti_colormap and get_kitti360_colormap are imported here if needed!


def get_grid_coords(dims, resolution):
    """
    :param dims: the dimensions of the grid [x, y, z] (i.e. [256, 256, 32])
    :return coords_grid: is the center coords of voxels in the grid
    """

    g_xx = np.arange(0, dims[0]) 
    g_yy = np.arange(0, dims[1]) 
    g_zz = np.arange(0, dims[2]) 

    # Obtaining the grid with coords...
    xx, yy, zz = np.meshgrid(g_xx, g_yy, g_zz)
    coords_grid = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T
    coords_grid = coords_grid.astype(np.float32)
    resolution = np.array(resolution, dtype=np.float32).reshape([1, 3])

    coords_grid = (coords_grid * resolution) + resolution / 2

    return coords_grid

def save_occ(
        save_dir, 
        gaussian, 
        name,
        sem=False,
        cap=2,
        dataset='nusc'
    ):

    if dataset == 'nusc':
        voxel_size = [0.5] * 3
        vox_origin = [-50.0, -50.0, -5.0]
        vmin, vmax = 0, 16
    elif dataset == 'kitti':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 19
    elif dataset == 'kitti360':
        voxel_size = [0.2] * 3
        vox_origin = [0.0, -25.6, -2.0]
        vmin, vmax = 1, 18

    voxels = gaussian[0].cpu().to(torch.int)
    voxels[0, 0, 0] = 1
    voxels[-1, -1, -1] = 1
    if not sem:
        voxels[..., (-cap):] = 0
        for z in range(voxels.shape[-1] - cap):
            mask = (voxels > 0)[..., z]
            voxels[..., z][mask] = z + 1 
    
    # Compute the voxels coordinates
    grid_coords = get_grid_coords(
        voxels.shape, voxel_size
    ) + np.array(vox_origin, dtype=np.float32).reshape([1, 3])

    grid_coords = np.vstack([grid_coords.T, voxels.reshape(-1)]).T
    # Get the voxels inside FOV
    fov_grid_coords = grid_coords

    # Remove empty and unknown voxels
    if not sem:
        fov_voxels = fov_grid_coords[
            (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 100)
        ]
    else:
        if dataset == 'nusc':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] >= 0) & (fov_grid_coords[:, 3] < 17)
            ]
        elif dataset == 'kitti360':
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 19)
            ]
        else:
            fov_voxels = fov_grid_coords[
                (fov_grid_coords[:, 3] > 0) & (fov_grid_coords[:, 3] < 20)
            ]
    print(f"[Status] Rendering {len(fov_voxels)} occupancy voxels...")
    
    figure = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    
    # Draw occupied inside FOV voxels
    voxel_size = sum(voxel_size) / 3
    if not sem:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            colormap="jet",
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
        )
    else:
        plt_plot_fov = mlab.points3d(
            fov_voxels[:, 0],
            -fov_voxels[:, 1],
            fov_voxels[:, 2],
            fov_voxels[:, 3],
            scale_factor=1.0 * voxel_size,
            mode="cube",
            opacity=1.0,
            vmin=vmin,
            vmax=vmax, # 16
        )

    plt_plot_fov.glyph.scale_mode = "scale_by_vector"
    if sem:
        if dataset == 'nusc':
            colors = np.array(
                [
                    [  0,   0,   0, 255],       # others
                    [255, 120,  50, 255],       # barrier              orange
                    [255, 192, 203, 255],       # bicycle              pink
                    [255, 255,   0, 255],       # bus                  yellow
                    [  0, 150, 245, 255],       # car                  blue
                    [  0, 255, 255, 255],       # construction_vehicle cyan
                    [255, 127,   0, 255],       # motorcycle           dark orange
                    [255,   0,   0, 255],       # pedestrian           red
                    [255, 240, 150, 255],       # traffic_cone         light yellow
                    [135,  60,   0, 255],       # trailer              brown
                    [160,  32, 240, 255],       # truck                purple                
                    [255,   0, 255, 255],       # driveable_surface    dark pink
                    [139, 137, 137, 255],       # other_flat
                    [ 75,   0,  75, 255],       # sidewalk             dark purple
                    [150, 240,  80, 255],       # terrain              light green          
                    [230, 230, 250, 255],       # manmade              white
                    [  0, 175,   0, 255],       # vegetation           green
                ]
            ).astype(np.uint8)
        elif dataset == 'kitti360':
            colors = (get_kitti360_colormap()[1:, :] * 255).astype(np.uint8)
        else:
            colors = (get_kitti_colormap()[1:, :] * 255).astype(np.uint8)

        plt_plot_fov.module_manager.scalar_lut_manager.lut.table = colors

    # =========================================================================
    # VIEWING AND SAVING LOGIC
    # =========================================================================
    # --- STABILIZATION BLOCK ---
    # 8 invisible corners to force a perfect 100x100x10 symmetric bounding box
    b_x = [-50,  50, 50, -50, -50,  50, 50, -50]
    b_y = [-50, -50, 50,  50, -50, -50, 50,  50]
    b_z = [-5,  -5, -5,  -5,   5,   5,  5,   5]
    mlab.points3d(b_x, b_y, b_z, scale_factor=1e-5, color=(1,1,1))

    # Nudge the camera target to push the scene UP and RIGHT into the perfect center
    mlab.view(azimuth=150, elevation=70, distance=180, focalpoint=[0, 0, 0])
    mlab.draw() 

    filepath = os.path.join(save_dir, f'{name}.png')
    
    if offscreen:
        mlab.savefig(filepath)
        mlab.close(figure_mlab if 'figure_mlab' in locals() else figure)
        print(f"[Status] Saved offline snapshot to {filepath}")
    else:
        mlab.savefig(filepath)
        print(f"[Status] Saved snapshot to {filepath}. Launching interactive window...")
        mlab.show()
        mlab.close(figure_mlab if 'figure_mlab' in locals() else figure)


def get_nuscenes_colormap():
    colors = np.array(
        [
            [  0,   0,   0, 255],       # others
            [255, 120,  50, 255],       # barrier              orange
            [255, 192, 203, 255],       # bicycle              pink
            [255, 255,   0, 255],       # bus                  yellow
            [  0, 150, 245, 255],       # car                  blue
            [  0, 255, 255, 255],       # construction_vehicle cyan
            [255, 127,   0, 255],       # motorcycle           dark orange
            [255,   0,   0, 255],       # pedestrian           red
            [255, 240, 150, 255],       # traffic_cone         light yellow
            [135,  60,   0, 255],       # trailer              brown
            [160,  32, 240, 255],       # truck                purple                
            [255,   0, 255, 255],       # driveable_surface    dark pink
            [139, 137, 137, 255],       # other_flat
            [ 75,   0,  75, 255],       # sidewalk             dard purple
            [150, 240,  80, 255],       # terrain              light green          
            [230, 230, 250, 255],       # manmade              white
            [  0, 175,   0, 255],       # vegetation           green
        ]
    ).astype(np.float32) / 255.
    return colors

def save_gaussian(save_dir, gaussian, name, scalar=1.5, ignore_opa=False, filter_zsize=False):

    empty_label = 17
    sem_cmap = get_nuscenes_colormap()

    torch.save(gaussian, os.path.join(save_dir, f'{name}_attr.pth'))

    means = gaussian.means[0].detach().cpu().numpy() # g, 3
    scales = gaussian.scales[0].detach().cpu().numpy() # g, 3
    rotations = gaussian.rotations[0].detach().cpu().numpy() # g, 4
    opas = gaussian.opacities[0]
    if opas.numel() == 0:
        opas = torch.ones_like(gaussian.means[0][..., :1])
    opas = opas.squeeze().detach().cpu().numpy() # g
    sems = gaussian.semantics[0].detach().cpu().numpy() # g, 18
    pred = np.argmax(sems, axis=-1)

    if ignore_opa:
        opas[:] = 1.
        mask = (pred != empty_label)
    else:
        mask = (pred != empty_label) & (opas > 0.75)

    if filter_zsize:
        zdist, zbins = np.histogram(means[:, 2], bins=100)
        zidx = np.argsort(zdist)[::-1]
        for idx in zidx[:10]:
            binl = zbins[idx]
            binr = zbins[idx + 1]
            zmsk = (means[:, 2] < binl) | (means[:, 2] > binr)
            mask = mask & zmsk
        
        z_small_mask = scales[:, 2] > 0.1
        mask = z_small_mask & mask

    means = means[mask]
    scales = scales[mask]
    rotations = rotations[mask]
    opas = opas[mask]
    pred = pred[mask]

    ellipNumber = means.shape[0]

    # =========================================================================
    # MAYAVI RENDERING ENGINE (Runs Universally, C++ Vectorized)
    # =========================================================================
    figure_mlab = mlab.figure(size=(2560, 1440), bgcolor=(1, 1, 1))
    figure_mlab.scene.disable_render = True 

    # Safety cap for memory
    max_gaussians = 50000 
    
    if ellipNumber > max_gaussians:
        print(f"[Status] Decimating {ellipNumber} Gaussians down to {max_gaussians} for rendering...")
        idx = np.random.choice(ellipNumber, max_gaussians, replace=False)
        means_m = means[idx]
        scales_m = scales[idx]
        rotations_m = rotations[idx]
        opas_m = opas[idx]
        pred_m = pred[idx]
        render_count = max_gaussians
    else:
        means_m = means
        scales_m = scales
        rotations_m = rotations
        opas_m = opas
        pred_m = pred
        render_count = ellipNumber

    print(f"[Status] Vectorizing 3D geometry for {render_count} Gaussians using C++ TensorGlyphs...")

    # 1. Swap X and Y to match Occupancy Grid Orientation
    means_aligned = np.zeros_like(means_m)
    means_aligned[:, 0] = means_m[:, 1]  
    means_aligned[:, 1] = -means_m[:, 0] 
    means_aligned[:, 2] = means_m[:, 2]  

    # 2. Vectorize Covariance Matrices
    rot_mats = np.array([Quaternion(r).rotation_matrix.T for r in rotations_m])
    
    S_mats = np.zeros((render_count, 3, 3))
    S_mats[:, 0, 0] = scales_m[:, 0] * scalar
    S_mats[:, 1, 1] = scales_m[:, 1] * scalar
    S_mats[:, 2, 2] = scales_m[:, 2] * scalar
    
    covs = rot_mats @ S_mats @ rot_mats.transpose(0, 2, 1)

    # 3. Apply the 90-degree spatial rotation to the Covariance Tensors
    T = np.array([
        [ 0,  1,  0], 
        [-1,  0,  0], 
        [ 0,  0,  1]
    ], dtype=np.float32)
    
    covs = T @ covs @ T.T

    # 4. Build the C++ VTK PolyData Array
    pts = tvtk.PolyData(points=means_aligned)
    pts.point_data.tensors = covs.reshape(-1, 9)
    pts.point_data.tensors.name = 'tensors'
    
    # Cast to float32 to prevent the "All Blue" VTK bug
    pts.point_data.scalars = pred_m.astype(np.float32)
    pts.point_data.scalars.name = 'semantics'

    src = mlab.pipeline.add_dataset(pts)
    sphere = tvtk.SphereSource(radius=1.0, phi_resolution=6, theta_resolution=6)
    
    tg = tvtk.TensorGlyph(
        scale_factor=1.0,
        extract_eigenvalues=True,
        color_glyphs=True,
        color_mode='scalars' 
    )
    tg.set_source_connection(sphere.output_port)
    
    glyph_filter = mlab.pipeline.user_defined(src, filter=tg)
    surf = mlab.pipeline.surface(glyph_filter)

    # Enforce exact NuScenes Colormap mapping
    lut = surf.module_manager.scalar_lut_manager.lut.table.to_array()
    n_colors = len(sem_cmap)
    lut[:n_colors, :3] = (sem_cmap * 255).astype(np.uint8)[:, :3]
    lut[:n_colors, 3] = 255 
    
    surf.module_manager.scalar_lut_manager.lut.table = lut
    surf.module_manager.scalar_lut_manager.use_default_range = False
    surf.module_manager.scalar_lut_manager.data_range = [0.0, 255.0]

    figure_mlab.scene.disable_render = False

    # =========================================================================
    # VIEWING AND SAVING LOGIC
    # =========================================================================
    # --- STABILIZATION BLOCK ---
    # 8 invisible corners to force a perfect 100x100x10 symmetric bounding box
    b_x = [-50,  50, 50, -50, -50,  50, 50, -50]
    b_y = [-50, -50, 50,  50, -50, -50, 50,  50]
    b_z = [-5,  -5, -5,  -5,   5,   5,  5,   5]
    mlab.points3d(b_x, b_y, b_z, scale_factor=1e-5, color=(1,1,1))

    # Nudge the camera target to push the scene UP and RIGHT into the perfect center
    mlab.view(azimuth=150, elevation=70, distance=180, focalpoint=[0, 0, 0])
    mlab.draw() 

    filepath = os.path.join(save_dir, f'{name}.png')
    
    if offscreen:
        mlab.savefig(filepath)
        mlab.close(figure_mlab if 'figure_mlab' in locals() else figure)
        print(f"[Status] Saved offline snapshot to {filepath}")
    else:
        mlab.savefig(filepath)
        print(f"[Status] Saved snapshot to {filepath}. Launching interactive window...")
        mlab.show()
        mlab.close(figure_mlab if 'figure_mlab' in locals() else figure)

def save_gaussian_topdown(save_dir, anchor_init, gaussian, name):
    init_means = safe_sigmoid(anchor_init[:, :2]) * 100 - 50
    means = [init_means] + [g.means[0, :, :2] for g in gaussian]

    plt.clf(); plt.cla()
    fig = plt.figure(figsize=(24., 16.))
    grid = ImageGrid(fig, 111,  
                    nrows_ncols=(1, 5),  
                    axes_pad=0.,  
                    share_all=True
                    )
    grid[0].get_yaxis().set_ticks([])
    grid[0].get_xaxis().set_ticks([])
    for ax, im in zip(grid, means):
        im = im.cpu()
        ax.scatter(im[:, 0], im[:, 1], s=0.1, marker='o')
    plt.savefig(os.path.join(save_dir, f"{name}.jpg"))
    plt.close(fig)