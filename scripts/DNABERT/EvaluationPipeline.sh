pip install --no-index -r ../../requirements.txt
echo 'Libraries Installed'


python genus_kNN.py --input_path=../../data -k 4 --model dnabert --checkpoint dnabert/4-new-12w-0
python genus_kNN.py --input_path=../../data -k 5 --model dnabert --checkpoint dnabert/5-new-12w-0
python genus_kNN.py --input_path=../../data -k 6 --model dnabert --checkpoint dnabert/6-new-12w-0


python Linear_probing.py --input_path=../../data -k 4 --model dnabert --checkpoint dnabert/4-new-12w-0
python Linear_probing.py --input_path=../../data -k 5 --model dnabert --checkpoint dnabert/5-new-12w-0
python Linear_probing.py --input_path=../../data -k 6 --model dnabert --checkpoint dnabert/6-new-12w-0
