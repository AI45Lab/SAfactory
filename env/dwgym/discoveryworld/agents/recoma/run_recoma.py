import argparse
import json
import logging
from pathlib import Path

from dotenv import load_dotenv
from recoma.run_inference import build_configurable_systems

from agents.recoma import *

load_dotenv()  # take environment variables from .env.


def parse_arguments():
    arg_parser = argparse.ArgumentParser(description='Run inference')
    arg_parser.add_argument('--config', type=str, required=True, help="Model and Inference config")
    arg_parser.add_argument('--debug', default=False, help="Debug mode", action="store_true")
    arg_parser.add_argument('--output_dir', type=str, default="output_dir/recoma", help="Output directory")
    return arg_parser.parse_args()

def create_prediction_alldata(example_prediction):
    metadata_json = {}
    try:
        pred_json = json.loads(example_prediction.prediction)
        if not isinstance(pred_json, list) and not isinstance(pred_json, dict):
            pred_json = example_prediction.prediction
        elif isinstance(pred_json, dict):
            # check for metadata and answers
            if "metadata" in pred_json:
                metadata_json = pred_json.pop("metadata")
            if "answer" in pred_json:
                pred_json = pred_json["answer"]
    except Exception:
        pred_json = example_prediction.prediction
    all_data_dict = example_prediction.example.__dict__
    all_data_dict["predicted"] = pred_json
    # Append any metadata from the final state
    if example_prediction.final_state and example_prediction.final_state.data:
        metadata_json = example_prediction.final_state.data | metadata_json
    if metadata_json:
        all_data_dict["metadata"] = metadata_json
    return pred_json, all_data_dict


def dump_predictions(example_predictions, output_dir):
    # Dump Predictions
    with open(output_dir + "/predictions.json", "w") as output_fp, \
            open(output_dir + "/all_data.jsonl", "w") as all_data_fp:
        prediction_dump = {}
        for x in example_predictions:
            pred_json, all_data_dict = create_prediction_alldata(x)
            all_data_fp.write(json.dumps(all_data_dict) + "\n")
            prediction_dump[x.example.unique_id] = pred_json
        json.dump(prediction_dump, output_fp)


def main():
    parsed_args = parse_arguments()
    output_dir = parsed_args.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    logging.basicConfig(level=logging.ERROR)
    if parsed_args.debug:
        logging.getLogger('recoma').setLevel(level=logging.DEBUG)
        logging.getLogger('agents.recoma').setLevel(level=logging.DEBUG)
        logging.getLogger('agents.recoma').setLevel(level=logging.DEBUG)
    example_predictions = []
    config_sys = build_configurable_systems(parsed_args.config, output_dir)
    reader = config_sys.reader
    search_algo = config_sys.search

    with open(output_dir + "/source_config.json", "w") as output_fp:
        output_fp.write(json.dumps(config_sys.source_json, indent=2))
    # Examples are populated by the dataset name in the config file
    for example in reader.get_examples(None):
        example_predictions.append(search_algo.predict(example))
        _, all_data_dict = create_prediction_alldata(example_predictions[-1])
        with open(output_dir + f"/{example.unique_id}_data.json", "w") as output_fp:
            json.dump(all_data_dict, output_fp, indent=2)


    dump_predictions(example_predictions, output_dir=output_dir)


if __name__ == "__main__":
    main()
