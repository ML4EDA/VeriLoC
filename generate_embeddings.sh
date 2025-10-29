export HUGGING_FACE_TOKEN='your_token'

# Navigate to the script directory to ensure relative paths work correctly
cd "$(dirname "$0")"

#generate module level embeddings
python src/generate_clverilog_embeddings.py --embedding_type "module"

#generate line level embeddings
python src/generate_clverilog_embeddings.py --embedding_type "line"