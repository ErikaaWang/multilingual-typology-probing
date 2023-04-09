for CORPUS in $(cat scripts/languages_common.lst); do
  echo "python preprocess_treebank.py $CORPUS --bloom bloom-560m" --use-gpu
  python preprocess_treebank.py $CORPUS --bloom bloom-560m --use-gpu
done