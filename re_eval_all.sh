for f in *; do
  echo "File -> $f"
  python run.py --type evaluate --cfg_file configs/city_rcnn_dance.yaml det_model "rcnn_dance/try100xloss/{$f}" > "tmp/{$f}.txt"
done