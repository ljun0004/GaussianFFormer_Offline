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
    # 1. GLOBAL SETTINGS & DDP INITIALIZATION (torchrun native)
    # =========================================================================
    set_random_seed(args.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    cfg = Config.fromfile(args.py_config)
    cfg.work_dir = args.work_dir

    if args.gpus > 1:
        distributed = True
        # Rely on torchrun's environment variables for master/port/rank setup
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
        
        world_size = dist.get_world_size()
        cfg.gpu_ids = range(world_size)
        torch.cuda.set_device(local_rank)

        if local_rank != 0:
            import builtins
            builtins.print = pass_print
    else:
        distributed = False
        world_size = 1
    
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(args.work_dir, f'{timestamp}_eval_offline.log')
    logger = MMLogger('selfocc', log_file=log_file)
    MMLogger._instance_dict['selfocc'] = logger
    logger.info(f'Config:\n{cfg.pretty_text}')

    # =========================================================================
    # 2. BUILD MODEL & METRICS
    # =========================================================================
    import model
    from dataset import get_dataloader

    my_model = build_segmentor(cfg.model)
    my_model.init_weights()
    
    # =========================================================================
    # MULTI-GPU EVALUATION FIX: No DDP wrapping needed for pure inference
    # =========================================================================
    my_model = my_model.cuda()
    raw_model = my_model

    # Force batch size to 1 to match offline extraction sequences
    cfg.train_loader['batch_size'] = 1  
    cfg.train_loader['num_workers'] = 8  
    cfg.train_loader['shuffle'] = False 

    # Load the TRAIN dataset loader!
    train_dataset_loader, _ = get_dataloader(
        cfg.train_dataset_config,
        cfg.val_dataset_config,
        cfg.train_loader,
        cfg.val_loader,
        dist=distributed,
        iter_resume=False,
        train_sampler_config=dict(shuffle=False, drop_last=False),
        val_only=False)

    from misc.metric_util import MeanIoU
    miou_metric = MeanIoU(
        list(range(1, 17)), 17, 
        ['barrier', 'bicycle', 'bus', 'car', 'construction_vehicle',
         'motorcycle', 'pedestrian', 'traffic_cone', 'trailer', 'truck',
         'driveable_surface', 'other_flat', 'sidewalk', 'terrain', 'manmade',
         'vegetation'],
         True, 17, filter_minmax=False)
    miou_metric.reset()

    my_model.eval()
    os.environ['eval'] = 'true'

    # =========================================================================
    # 3. FRAME-BY-FRAME EVALUATION LOOP
    # =========================================================================
    total_frames_local = len(train_dataset_loader)
    total_frames_global = total_frames_local * world_size
    
    # Initialize a local counter for this specific GPU
    local_missing_checkpoints = 0
    
    # Dynamically build the scenes directory based on work_dir
    scenes_dir = osp.join(args.work_dir, 'scenes')

    if local_rank == 0:
        logger.info(f"Starting Evaluation on Extracted MAP Checkpoints.")
        logger.info(f"Target Checkpoint Directory: {scenes_dir}")

    with torch.no_grad():
        for i_iter, data in enumerate(train_dataset_loader):
            global_scene_id = i_iter * world_size + local_rank
            real_scene_id = data['sample_idx'][0]
            
            # Look for the checkpoint inside the dynamically built scenes_dir
            load_path = osp.join(scenes_dir, f'idx_{global_scene_id}_token_{real_scene_id}.pth')

            if not osp.exists(load_path):
                local_missing_checkpoints += 1
                if local_rank == 0:
                    logger.warning(f"Missing checkpoint for Global Frame {global_scene_id} (ID: {real_scene_id}). Skipping.")
                continue

            ckpt = torch.load(load_path, map_location='cuda')
            x_bar = ckpt['x_bar'].cuda()

            for k in list(data.keys()):
                if isinstance(data[k], torch.Tensor):
                    data[k] = data[k].cuda()
            
            _ = data.pop('img')
            _ = data.pop('sample_idx')

            with torch.autocast(device_type='cuda', dtype=torch.float16):
                result_dict = my_model(x_bar=x_bar, metas=data)

            if 'final_occ' in result_dict:
                for idx, pred in enumerate(result_dict['final_occ']):
                    pred_occ = pred
                    gt_occ = result_dict['sampled_label'][idx]
                    occ_mask = result_dict['occ_mask'][idx].flatten()
                    
                    # 1. Add to the Global tracker (for the final paper score)
                    miou_metric._after_step(pred_occ, gt_occ, occ_mask)
                    
                    # 2. OPTIONAL: Calculate local frame metric for logging
                    if i_iter % cfg.print_freq == 0 and local_rank == 0:
                        # Create a temporary, local-only tracker just for this frame
                        temp_metric = MeanIoU(list(range(1, 17)), 17, [], True, 17, filter_minmax=False)
                        
                        # ---> THE MISSING LINE <---
                        temp_metric.reset() 
                        temp_metric._after_step(pred_occ, gt_occ, occ_mask)
                        
                        # Manually do the math to avoid DDP sync barriers
                        ious = []
                        for i in range(16):
                            if temp_metric.total_seen[i] > 0:
                                iou = temp_metric.total_correct[i] / (temp_metric.total_seen[i] + temp_metric.total_positive[i] - temp_metric.total_correct[i])
                                ious.append(iou.item())
                        
                        frame_miou = np.mean(ious) * 100 if ious else 0.0
                        logger.info(f'[EVAL] Global Frame {global_scene_id}/{total_frames_global} | Local Frame {i_iter}/{total_frames_local} | Frame mIoU: {frame_miou:.2f}%')
                    
    miou, iou2 = miou_metric._after_epoch()

    # =========================================================================
    # 4. MULTI-GPU SYNCHRONIZATION FOR MISSING FRAMES
    # =========================================================================
    # Sum the missing frames across all GPUs to get the true total
    missing_tensor = torch.tensor([local_missing_checkpoints], dtype=torch.int32, device='cuda')
    if distributed:
        dist.all_reduce(missing_tensor, op=dist.ReduceOp.SUM)
    global_missing = missing_tensor.item()
    
    if local_rank == 0:
        logger.info("===================================================")
        logger.info(f"Final Evaluation Results on {total_frames_global - global_missing} extracted frames:")
        logger.info(f"Frames skipped globally due to missing/NaN checkpoints: {global_missing}")
        logger.info(f"mIoU:  {miou:.4f}")
        logger.info(f"IoU2:  {iou2:.4f}")
        logger.info("===================================================")
        
    miou_metric.reset()

    # Cleanly tear down the distributed process group
    if distributed:
        dist.destroy_process_group()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Offline MAP Evaluation Script')
    parser.add_argument('--py-config', default='config/tpv_lidarseg.py')
    # Default work_dir is set here, scenes will be automatically inferred as work_dir/scenes
    parser.add_argument('--work-dir', type=str, default='./out/offline_eval')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # -------------------------------------------------------------------------
    # LAUNCH MODIFICATION: Rely on torchrun instead of multiprocessing.spawn
    # -------------------------------------------------------------------------
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    args.gpus = torch.cuda.device_count()
    
    if local_rank == 0:
        print(args)

    main(local_rank, args)