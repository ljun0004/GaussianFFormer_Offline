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

    my_model = build_segmentor(cfg.model).cuda()
    my_model.eval()  
    
    for p in my_model.parameters():
        p.requires_grad = False

    cfg.train_loader['batch_size'] = 1  
    cfg.train_loader['num_workers'] = 8  
    cfg.train_loader['shuffle'] = False 

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
    optimization_steps = 2000   
    learning_rate = 0.005

    total_frames_local = len(train_dataset_loader)
    total_frames_global = total_frames_local * world_size

    if local_rank == 0:
        logger.info(f"Starting Inverse-Graphics Offline Extraction.")
        logger.info(f"Total Frames per GPU: {total_frames_local} | Total Global Frames: {total_frames_global}")

    # Pre-create the directory so all GPU threads have a valid path to verify against
    ckpt_dir = osp.join(args.work_dir, 'ckpts') 
    if local_rank == 0:
        os.makedirs(ckpt_dir, exist_ok=True)

    for i_iter, data in enumerate(train_dataset_loader):
        global_scene_id = i_iter * world_size + local_rank
        current_frame = i_iter + 1

        # ---------------------------------------------------------------------
        # MODIFICATION 1: CHECK FOR EXISTING SCENE BEFORE ALLOCATING VRAM
        # ---------------------------------------------------------------------
        # Extract token early without destroying the original dictionary structure
        real_scene_id = data['sample_idx'][0]
        save_path = osp.join(ckpt_dir, f'idx_{global_scene_id}_token_{real_scene_id}.pth')

        if osp.exists(save_path):
            if local_rank == 0:
                logger.info(f"  -> Skipping Frame [{current_frame}/{total_frames_local}] (ID: {real_scene_id}) - Checkpoint already exists.")
            continue

        # 1. Push structural ground truth tokens and images to device
        for k in list(data.keys()):
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].cuda()
        
        input_imgs = data.pop('img')
        real_scene_id = data.pop('sample_idx')[0] 
        
        # ---------------------------------------------------------------------
        # PHASE 1: THE WARM START 
        # ---------------------------------------------------------------------
        time_s = time.time()
        
        x_bar = my_model.get_warm_start(imgs=input_imgs, metas=data)
        optimizer = torch.optim.Adam([x_bar], lr=learning_rate)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        #             optimizer, 
        #             T_max=optimization_steps, 
        #             eta_min=1e-4
        #         )

        if local_rank == 0:
            try:
                num_anchors = cfg.model.lifter.num_anchor
                num_random = cfg.model.lifter.random_samples
            except AttributeError:
                num_anchors, num_random = "Unknown", "Unknown"

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
        
        # MODIFICATION 2A: Initialize trackers for the lowest loss state
        best_loss = float('inf')
        best_x_bar = None
        
        scaler = torch.cuda.amp.GradScaler()

        for step in range(optimization_steps):
            optimizer.zero_grad()

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result_dict = my_model(x_bar=x_bar, metas=data)

                loss_input = {'metas': data}
                for loss_input_key, loss_input_val in cfg.loss_input_convertion.items():
                    loss_input[loss_input_key] = result_dict[loss_input_val]

                loss, loss_dict = loss_func(loss_input)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()

            current_loss = loss.item()

            # -----------------------------------------------------------------
            # MODIFICATION 2B: RECORD THE BEST TENSOR WEIGHTS
            # -----------------------------------------------------------------
            if current_loss < best_loss:
                best_loss = current_loss
                # .clone().detach().cpu() safely locks the weights into RAM 
                # without polluting VRAM or keeping the gradient graph alive
                best_x_bar = x_bar.clone().detach().cpu() 

            # Logging Telemetry
            if step % 100 == 0 and local_rank == 0:
                current_lr = optimizer.param_groups[0]['lr']
                detailed_loss = ', '.join([f'{k}: {v:.4f}' for k, v in loss_dict.items()])
                logger.info(f'  [Frame {current_frame}/{total_frames_local}] [Step {step:4d}/{optimization_steps}] LR: {current_lr:.6f} | Patience Counter: {patience_counter:3d} | Total Loss: {current_loss:.4f} | {detailed_loss}')                
                
                if writer is not None:
                    writer.add_scalar(f'Loss_Scene_{global_scene_id}/total_loss', current_loss, step)
                    writer.add_scalar(f'Loss_Scene_{global_scene_id}/learning_rate', current_lr, step)
                    for k, v in loss_dict.items():
                        writer.add_scalar(f'Loss_Scene_{global_scene_id}/{k}', v, step)

            # Early Stopping Trigger
            if abs(previous_loss - current_loss) < 1e-4:
                patience_counter += 1
                if patience_counter >= 100:
                    if local_rank == 0:
                        logger.info(f"  --> Early stopping triggered at step {step}. MAP estimate stabilized.")
                    break
            else:
                patience_counter = 0
            previous_loss = current_loss
            
        time_e = time.time()
        
        # ---------------------------------------------------------------------
        # PHASE 3: ASSET EXPORT
        # ---------------------------------------------------------------------
        # MODIFICATION 2C: Save the lowest-loss snapshot, not the current one
        export_payload = {
            'token': real_scene_id,                
            'global_id': global_scene_id,          
            'x_bar': best_x_bar,                   # Now saving the best recorded state
        }

        torch.save(export_payload, save_path)
        
        if local_rank == 0:
            logger.info(f'Frame [{current_frame}/{total_frames_local}] (ID: {real_scene_id}) converged in {time_e - time_s:.2f}s with Lowest Loss: {best_loss:.4f}. Saved -> {save_path}')    
    
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