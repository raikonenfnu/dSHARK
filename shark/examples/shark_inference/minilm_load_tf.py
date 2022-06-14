import tensorflow as tf
from transformers import BertModel, BertTokenizer, TFBertModel
from shark.shark_inference import SharkInference
from shark.parser import parser
from urllib import request
import os

parser.add_argument(
    "--download_mlir_path",
    type=str,
    default="minilm_tf_inference.mlir",
    help="Specifies path to target mlir file that will be loaded.")
load_args, unknown = parser.parse_known_args()

MAX_SEQUENCE_LENGTH = 512

if __name__ == "__main__":
    # Prepping Data
    tokenizer = BertTokenizer.from_pretrained(
        "microsoft/MiniLM-L12-H384-uncased")
    text = "Replace me by any text you'd like."
    encoded_input = tokenizer(text,
                              padding='max_length',
                              truncation=True,
                              max_length=MAX_SEQUENCE_LENGTH)
    for key in encoded_input:
        encoded_input[key] = tf.expand_dims(
            tf.convert_to_tensor(encoded_input[key]), 0)
    file_link = "https://storage.googleapis.com/shark_tank/users/stanley/minilm_tf_inference.mlir"
    response = request.urlretrieve(file_link, load_args.download_mlir_path)
    if not os.path.isfile(load_args.download_mlir_path):
        raise ValueError(f"Tried looking for target mlir in {load_args.download_mlir_path}, but cannot be found.")
    with open(load_args.download_mlir_path, "rb") as input_file:
        minilm_mlir = input_file.read()
    shark_module = SharkInference(
        minilm_mlir,
        (encoded_input["input_ids"], encoded_input["attention_mask"],
         encoded_input["token_type_ids"]))
    shark_module.set_frontend("mhlo")
    shark_module.compile()

    print(
        shark_module.forward(
            (encoded_input["input_ids"], encoded_input["attention_mask"],
             encoded_input["token_type_ids"])))
