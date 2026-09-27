# python collector1.py \
#   --host 127.0.0.1 \
#   --port 5555 \
#   --num-successes 100 \
#   --max-attempts 1000 \
#   --position-scale 0.05 \
#   --output-dir ./matterix_beaker_dataset \
#   --grasp-z-offset 0.02 \
#   --max-steps 300 \
#   --with-state
#   # --save-debug-video \
#   # --debug-video-fps 20 


python collector2.py \
  --host 127.0.0.1 \
  --port 5555 \
  --num-successes 100 \
  --max-attempts 1000 \
  --position-scale 0.05 \
  --output-dir ./matterix_beaker_dataset_mixed_front \
  --beaker-grasp-z-offset 0.02 \
  --cylinder-grasp-z-offset 0.06 \
  --max-steps 300 \
  --with-state
  # --save-debug-video \
  # --debug-video-fps 20 

