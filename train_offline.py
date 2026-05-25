import time, argparse, os.path as osp, os
import torch, numpy as np
import torch.distributed as dist

from mmengine import Config
from mmengine.runner import set_random_seed
from mmengine.logging import MMLogger
from mmseg.models import build_segmentor

import warnings
warnings.filterwarnings("ignore")

def pass_print(*args, **kwargs):
    pass

def main(local_rank, args):
    # =========================================================================
    # 1. GLOBAL SETTINGS & DDP INITIALIZATION
    # =========================================================================
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    if args.gpus > 1:
        distributed = True
        ip = os.environ.get("MASTER_ADDR", "127.0.0.1")
        port = os.environ.get("MASTER_PORT", "20507")
        hosts = int(os.environ.get("WORLD_SIZE", 1))
        rank = int(os.environ.get("RANK", 0))
        gpus = torch.cuda.device_count()
        dist.init_process_group(
            backend="nccl", init_method=f"tcp://{ip}:{port}", 
            world_size=hosts * gpus, rank=rank * gpus + local_rank)
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    if local_rank == 0:
        os.makedirs(args.work_dir, exist_ok=True)
        os.makedirs(osp.join(args.work_dir, 'scenes'), exist_ok=True)
        cfg.dump(osp.join(args.work_dir, osp.basename(args.py_config)))
        
        # ADDED: Initialize TensorBoard writer
        from misc.tb_wrapper import WrappedTBWriter
        writer = WrappedTBWriter('selfocc', log_dir=osp.join(args.work_dir, 'tf'))
        WrappedTBWriter._instance_dict['selfocc'] = writer
    else:
        writer = None
        
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}_inverse_graphics.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Loaded Configuration:\n{cfg.pretty_text}')

    # =========================================================================
    # 2. BUILD MODEL ENGINE & EXTRACT LOSS
    # =========================================================================
    import model
    from dataset import get_dataloader
    from loss import OPENOCC_LOSS

    # We build the model but DO NOT wrap it in DistributedDataParallel (DDP).
    # We are optimizing scene parameters locally on each GPU, not syncing network weights.
    my_model = build_segmentor(cfg.model).cuda()
    my_model.eval()  # Permanently freeze dropout and batch norms
    
    for p in my_model.parameters():
        p.requires_grad = False

    # Force the train loader to be completely sequential (No Shuffling)
    cfg.train_loader['batch_size'] = 1  # Process one scene at a time for MAP optimization
    cfg.train_loader['num_workers'] = 8  # Adjust based on your CPU capabilities
    cfg.train_loader['shuffle'] = False # Critical for ensuring each GPU processes a unique scene without overlap

    # The dataloader's DistributedSampler automatically assigns different scenes to different GPUs    
    train_dataset_loader, val_dataset_loader = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=False,
        train_sampler_config=dict(shuffle=False, drop_last=False),
        val_only=False)

    loss_func = OPENOCC_LOSS.build(cfg.loss).cuda()

    # =========================================================================
    # 3. PER-SCENE MAP OPTIMIZATION LOOP
    # =========================================================================
    optimization_steps = 4000   
    learning_rate = 0.005

    total_frames_local = len(train_dataset_loader)
    total_frames_global = total_frames_local * world_size

    if local_rank == 0:
        logger.info(f"Starting Inverse-Graphics Offline Extraction.")
        logger.info(f"Total Frames per GPU: {total_frames_local} | Total Global Frames: {total_frames_global}")

    for i_iter, data in enumerate(train_dataset_loader):
        global_scene_id = i_iter * world_size + local_rank
        current_frame = i_iter + 1

        # 1. Push structural ground truth tokens and images to device
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()
        
        # Pop the image tensor explicitly as dictated by the original data flow
        input_imgs = data.pop('img')
        
        real_scene_id = data.pop('sample_idx')[0]
        
        # ---------------------------------------------------------------------
        # PHASE 1: THE WARM START (Bypassing the Encoder)
        # ---------------------------------------------------------------------
        time_s = time.time()
        
        # OPTION A: Use the Distribution-Based Initializer (Warm Start)
        x_bar = my_model.get_warm_start(imgs=input_imgs, metas=data)
        # OPTION B: Bypass the lifter for a purely mathematical scatter (Cold Start)
        # x_bar = my_model.get_cold_start(imgs=input_imgs, metas=data)

        # Spin up an isolated optimizer exclusively for this scene's geometry
        optimizer = torch.optim.Adam([x_bar], lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #             optimizer, 
        #             T_max=optimization_steps, 
        #             eta_min=1e-4
        #         )
        
        if local_rank == 0:
            num_anchors = cfg.model.lifter.num_anchor
            num_random = cfg.model.lifter.random_samples

            logger.info(f"\n=======================================================")
            logger.info(f"Optimizing MAP Estimate | Frame Progress: [{current_frame}/{total_frames_local}] | nuScenes ID: {real_scene_id}")
            logger.info(f"Initialized Parameters Shape: {x_bar.shape}")
            logger.info(f"  -> Anchored Gaussians initialized: {num_anchors}")
            logger.info(f"  -> Random Gaussians initialized:   {num_random}")
            logger.info(f"=======================================================")

        # ---------------------------------------------------------------------
        # PHASE 2: INVERSE GRAPHICS GRADIENT DESCENT
        # ---------------------------------------------------------------------
        previous_loss = float('inf')
        patience_counter = 0
        
        scaler = torch.cuda.amp.GradScaler()

        for step in range(optimization_steps):
            optimizer.zero_grad()

            # Wrap the forward pass and loss computation in AMP
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result_dict = my_model(x_bar=x_bar, metas=data)

                loss_input = {'metas': data}
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    loss_input[loss_input_key] = result_dict[loss_input_val]

                loss, loss_dict = loss_func(loss_input)
            
            # Scale the loss and step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # Logging Telemetry & TensorBoard Visualization
            if step % 100 == 0 and local_rank == 0:
                # Extract the current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                
                detailed_loss = ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
                
                logger.info(f'  [Frame {current_frame}/{total_frames_local}] [Step {step:4d}/{optimization_steps}] LR: {current_lr:.6f} | Patience Counter: {patience_counter:3d} | Total Loss: {loss.item():.4f} | {detailed_loss}')                
                
                # Output visualizations to TensorBoard
                if writer is not None:
                    writer.add_scalar(f'Loss_Scene_{global_scene_id}/total_loss', loss.item(), step)
                    writer.add_scalar(f'Loss_Scene_{global_scene_id}/learning_rate', current_lr, step)
                    for k, v in loss_dict.items():
                        writer.add_scalar(f'Loss_Scene_{global_scene_id}/{k}', v, step)

            # Early Stopping Trigger
            if abs(previous_loss - loss.item()) < 5e-4:
                patience_counter += 1
                if patience_counter >= 100:
                    if local_rank == 0:
                        logger.info(f"  --> Early stopping triggered at step {step}. MAP estimate stabilized.")
                    break
            else:
                patience_counter = 0
            previous_loss = loss.item()
            
        time_e = time.time()
        
        # ---------------------------------------------------------------------
        # PHASE 3: ASSET EXPORT
        # ---------------------------------------------------------------------
        # Changed 'ckpts' to 'scenes' or keep 'ckpts' based on your folder preferences
        ckpt_dir = osp.join(args.work_dir, 'ckpts') 
        os.makedirs(ckpt_dir, exist_ok=True)

        # Package parameters and metadata together
        export_payload = {
            'token': real_scene_id,                # The definitive Sample Token (UUID)
            'global_id': global_scene_id,          # Keep this inside for training logs
            'x_bar': x_bar.detach().cpu(),         # Your optimized Gaussian tensor
        }

        # RECOMMENDED NAMING: Uses the token first, so your dataloader can find it instantly
        # save_path = osp.join(scenes_dir, f'scene_{real_scene_id}.pth')
        
        # ALTERNATIVE NAMING (If you want both visible on disk):
        save_path = osp.join(ckpt_dir, f'idx_{global_scene_id}_token_{real_scene_id}.pth')

        torch.save(export_payload, save_path)
        
        if local_rank == 0:
            logger.info(f'Frame [{current_frame}/{total_frames_local}] (ID: {real_scene_id}) converged in {time_e - time_s:.2f}s. Saved -> {save_path}')    
    
    if writer is not None:
        writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline MAP Pre-Optimization Script')
    parser.add_argument('--py-config', default='config/nuscenes_gs25600_offline.py')
    parser.add_argument('--work-dir', type=str, default='./out/offline_map_extraction')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='nuscenes')
    args = parser.parse_args()
    
    ngpus = torch.cuda.device_count()
    args.gpus = ngpus

    if int(os.environ.get('LOCAL_RANK', 0)) == 0:
        print(args)

    if ngpus > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=args.gpus)
    else:
        main(0, args)