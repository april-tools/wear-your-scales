# a script to preprocess and generate different segment length recordings

python preprocess_ds.py --time_alignment 0 --segment_length 8 --output_dir data/preprocessed/ta0_sl8 --overwrite

python preprocess_ds.py --time_alignment 0 --segment_length 16 --output_dir data/preprocessed/ta0_sl16 --overwrite

python preprocess_ds.py --time_alignment 0 --segment_length 32 --output_dir data/preprocessed/ta0_sl32 --overwrite

python preprocess_ds.py --time_alignment 0 --segment_length 64 --output_dir data/preprocessed/ta0_sl64 --overwrite

python preprocess_ds.py --time_alignment 0 --segment_length 128 --output_dir data/preprocessed/ta0_sl128 --overwrite

python preprocess_ds.py --time_alignment 0 --segment_length 256 --output_dir data/preprocessed/ta0_sl256 --overwrite

python preprocess_ds.py --time_alignment 0 --segment_length 512 --output_dir data/preprocessed/ta0_sl512 --overwrite

python preprocess_ds.py --time_alignment 0 --segment_length 1024 --output_dir data/preprocessed/ta0_sl1024 --overwrite
