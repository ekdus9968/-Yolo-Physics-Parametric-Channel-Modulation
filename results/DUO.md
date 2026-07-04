Model summary (fused): 93 layers, 25,842,076 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 4.6it/s 7.6s
                   all       1111      10517      0.824      0.748      0.809      0.607
           holothurian        490       1079      0.834      0.741      0.817      0.587
               echinus        967       7201      0.852      0.858      0.903      0.692
               scallop         78        217      0.734      0.546       0.62      0.453
              starfish        659       2020      0.876      0.847      0.896      0.695
Speed: 0.1ms preprocess, 2.9ms inference, 0.0ms loss, 0.9ms postprocess per image
Results saved to C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\runs\detect\runs\ppcm\DUO_baseline-3

Model summary (fused): 93 layers, 25,842,076 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 4.8it/s 7.3s
                   all       1111      10517      0.824      0.748      0.809      0.607
           holothurian        490       1079      0.834      0.741      0.817      0.587
               echinus        967       7201      0.852      0.858      0.903      0.692
               scallop         78        217      0.734      0.546       0.62      0.453
              starfish        659       2020      0.876      0.847      0.896      0.695
Speed: 0.1ms preprocess, 2.8ms inference, 0.0ms loss, 1.0ms postprocess per image
Results saved to C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\runs\ppcm\DUO_stage2

venv) PS C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm> python train_ppcm_yolo.py --data data/DUO/data.yaml --weights yolov8m.pt --stage1 1 --stage2 0 --dataset-name DUO
=== PPCM config: stage1 | dataset: DUO ===
[train] PPCM attached as submodule. learnable params: 1
[train] augmentation: ALL OFF (mosaic/color/flip) — depth 정합 보장
[train] (depth 캐시는 원본 기준 — augment off 라 정합 완벽 유지)
New https://pypi.org/project/ultralytics/8.4.87 available  Update with 'pip install -U ultralytics'
Ultralytics 8.4.86  Python-3.10.11 torch-2.12.1+cu132 CUDA:0 (NVIDIA GeForce RTX 5060 Ti, 8151MiB)
engine\trainer: agnostic_nms=False, amp=True, angle=1.0, augment=False, auto_augment=randaugment, batch=16, bgr=0.0, box=7.5, cache=False, cfg=None, classes=None, close_mosaic=10, cls=0.5, cls_pw=0.0, compile=False, conf=None, copy_paste=0.0, copy_paste_mode=flip, cos_lr=False, cutmix=0.0, data=data/DUO/data.yaml, degrees=0.0, deterministic=True, device=None, dfl=1.5, dis=6.0, distill_model=None, dnn=False, dropout=0.0, dynamic=False, embed=None, end2end=None, epochs=12, erasing=0.4, exist_ok=False, fliplr=0.0, flipud=0.0, format=torchscript, fraction=1.0, freeze=None, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, imgsz=640, iou=0.7, keras=False, kobj=1.0, line_width=None, lr0=0.01, lrf=0.01, mask_ratio=4, max_det=300, mixup=0.0, mode=train, model=yolov8m.pt, momentum=0.937, mosaic=0.0, multi_scale=0.0, name=DUO_stage1, nbs=64, nms=False, opset=None, optimize=False, optimizer=auto, overlap_mask=True, patience=100, perspective=0.0, plots=True, pose=12.0, pretrained=True, profile=False, project=C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\runs\ppcm, quantize=None, rect=False, resume=False, retina_masks=False, rle=1.0, save=True, save_conf=False, save_crop=False, save_dir=C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\runs\ppcm\DUO_stage1, save_frames=False, save_json=False, save_period=-1, save_txt=False, scale=0.0, seed=0, shear=0.0, show=False, show_boxes=True, show_conf=True, show_labels=True, simplify=True, single_cls=False, source=None, split=val, stream_buffer=False, task=detect, time=None, tracker=tracktrack.yaml, translate=0.0, val=True, verbose=True, vid_stride=1, visualize=False, warmup_bias_lr=0.1, warmup_epochs=3.0, warmup_momentum=0.8, weight_decay=0.0005, workers=8, workspace=None
Overriding model.yaml nc=80 with nc=4

                   from  n    params  module                                       arguments                     
  0                  -1  1      1392  ultralytics.nn.modules.conv.Conv             [3, 48, 3, 2]                 
  1                  -1  1     41664  ultralytics.nn.modules.conv.Conv             [48, 96, 3, 2]                
  2                  -1  2    111360  ultralytics.nn.modules.block.C2f             [96, 96, 2, True]             
  3                  -1  1    166272  ultralytics.nn.modules.conv.Conv             [96, 192, 3, 2]               
  4                  -1  4    813312  ultralytics.nn.modules.block.C2f             [192, 192, 4, True]           
  5                  -1  1    664320  ultralytics.nn.modules.conv.Conv             [192, 384, 3, 2]              
  6                  -1  4   3248640  ultralytics.nn.modules.block.C2f             [384, 384, 4, True]           
  7                  -1  1   1991808  ultralytics.nn.modules.conv.Conv             [384, 576, 3, 2]              
  8                  -1  2   3985920  ultralytics.nn.modules.block.C2f             [576, 576, 2, True]           
  9                  -1  1    831168  ultralytics.nn.modules.block.SPPF            [576, 576, 5]                 
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 12                  -1  2   1993728  ultralytics.nn.modules.block.C2f             [960, 384, 2]                 
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']          
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 15                  -1  2    517632  ultralytics.nn.modules.block.C2f             [576, 192, 2]                 
 16                  -1  1    332160  ultralytics.nn.modules.conv.Conv             [192, 192, 3, 2]              
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 18                  -1  2   1846272  ultralytics.nn.modules.block.C2f             [576, 384, 2]                 
 19                  -1  1   1327872  ultralytics.nn.modules.conv.Conv             [384, 384, 3, 2]              
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]                           
 21                  -1  2   4207104  ultralytics.nn.modules.block.C2f             [960, 576, 2]                 
 22        [15, 18, 21]  1   3778012  ultralytics.nn.modules.head.Detect           [4, 16, None, [192, 384, 576]]
Model summary: 170 layers, 25,858,636 parameters, 25,858,620 gradients, 79.1 GFLOPs

Transferred 469/475 items from pretrained weights
Freezing layer 'model.22.dfl.conv.weight'
AMP: running Automatic Mixed Precision (AMP) checks...
AMP: checks passed 
train: Fast image access  (ping: 0.10.0 ms, read: 1106.51204.9 MB/s, size: 450.7 KB)
train: Scanning C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\data\DUO\train\labels.cache... 6617 images, 54 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 6671/6671  0.0s
val: Fast image access  (ping: 0.00.0 ms, read: 1149.91102.6 MB/s, size: 479.3 KB)
val: Scanning C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\data\DUO\test\labels.cache... 1100 images, 11 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 1111/1111  0.0s
optimizer: 'optimizer=auto' found, ignoring 'lr0=0.01' and 'momentum=0.937' and determining best 'optimizer', 'lr0' and 'momentum' automatically... 
optimizer: AdamW(lr=0.00125, momentum=0.9) with parameter groups 77 weight(decay=0.0), 84 weight(decay=0.0005), 83 bias(decay=0.0)
Plotting labels to C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\runs\ppcm\DUO_stage1\labels.jpg... 
Image sizes 640 train, 640 val
Using 8 dataloader workers
Logging results to C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\runs\ppcm\DUO_stage1
Starting training for 12 epochs...

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       1/12      6.18G      1.072      1.102      1.061        126        640: 100% ━━━━━━━━━━━━ 417/417 3.4it/s 2:01
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.0it/s 7.0s
                   all       1111      10517      0.764      0.628      0.705      0.488

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       2/12      6.25G      0.909     0.7105      0.986        179        640: 100% ━━━━━━━━━━━━ 417/417 3.6it/s 1:55
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.2it/s 6.8s
                   all       1111      10517      0.755      0.685       0.74      0.507
Closing dataloader mosaic

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       3/12      6.33G     0.8561     0.6456     0.9666        172        640: 100% ━━━━━━━━━━━━ 417/417 3.7it/s 1:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.5it/s 6.4s
                   all       1111      10517      0.785      0.624      0.734      0.519

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       4/12      6.23G     0.7938     0.5735      0.939        212        640: 100% ━━━━━━━━━━━━ 417/417 3.9it/s 1:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.5it/s 6.4s
                   all       1111      10517        0.8       0.72      0.792      0.561

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       5/12      6.32G     0.7311     0.5097     0.9126        168        640: 100% ━━━━━━━━━━━━ 417/417 3.9it/s 1:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.5it/s 6.3s
                   all       1111      10517      0.834      0.696      0.801      0.579

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       6/12      6.28G     0.6885     0.4517     0.8952        172        640: 100% ━━━━━━━━━━━━ 417/417 3.9it/s 1:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.5it/s 6.4s
                   all       1111      10517      0.801      0.708      0.794      0.579

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       7/12      6.36G     0.6383     0.3998     0.8757        164        640: 100% ━━━━━━━━━━━━ 417/417 3.9it/s 1:48
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.3it/s 6.5s
                   all       1111      10517      0.847      0.687      0.787      0.577

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       8/12      6.24G     0.5906       0.36     0.8584        132        640: 100% ━━━━━━━━━━━━ 417/417 7.3s/it 50:38
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.3it/s 6.6s
                   all       1111      10517      0.847      0.698      0.797      0.591

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
       9/12      6.15G     0.5502     0.3245     0.8453        136        640: 100% ━━━━━━━━━━━━ 417/417 3.7it/s 1:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.5it/s 6.4s
                   all       1111      10517      0.825      0.747      0.809      0.607

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      10/12      6.34G     0.4986     0.2847     0.8291        192        640: 100% ━━━━━━━━━━━━ 417/417 3.7it/s 1:52
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.4it/s 6.5s
                   all       1111      10517      0.839      0.744      0.802        0.6

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      11/12      6.34G     0.4485     0.2566     0.8157        114        640: 100% ━━━━━━━━━━━━ 417/417 3.8it/s 1:49
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.7it/s 6.1s
                   all       1111      10517      0.848      0.715      0.792      0.596

      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      12/12      6.29G     0.3987     0.2305     0.8036        147        640: 100% ━━━━━━━━━━━━ 417/417 3.9it/s 1:47
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.7it/s 6.1s
                   all       1111      10517      0.864      0.713      0.789        0.6

Validating C:\Users\thisi\OneDrive\Documents\ERA-marin\ppcm\runs\ppcm\DUO_stage1\weights\best.pt...
Ultralytics 8.4.86  Python-3.10.11 torch-2.12.1+cu132 CUDA:0 (NVIDIA GeForce RTX 5060 Ti, 8151MiB)
Model summary (fused): 93 layers, 25,842,076 parameters, 0 gradients, 78.7 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 35/35 5.1it/s 6.9s
                   all       1111      10517      0.824      0.748      0.809      0.607
           holothurian        490       1079      0.834      0.741      0.817      0.587
               echinus        967       7201      0.852      0.858      0.903      0.692
               scallop         78        217      0.734      0.546       0.62      0.453
              starfish        659       2020      0.876      0.847      0.896      0.695