import os
from pathlib import Path

from flask import Flask, jsonify, request

from mlebench.grade import validate_submission
from mlebench.registry import registry

app = Flask(__name__)

from utils.config_mcts import load_cfg

cfg = load_cfg()
# PRIVATE_DATA_DIR = "/private/data"
base_dir = cfg.dataset_dir

def run_validation(submission: Path, competition_id: str) -> str:
    PRIVATE_DATA_DIR = Path(base_dir)
    new_registry = registry.set_data_dir(PRIVATE_DATA_DIR)
    competition = new_registry.get_competition(competition_id)
    print(competition.private_dir)
    is_valid, message = validate_submission(submission, competition)
    return is_valid, message


@app.route("/validate", methods=["POST"])
def validate():
    submission_file = request.files["file"]
    competition_id = request.headers.get('exp-id')
    print(f"competition_id is {competition_id}")
    submission_path = Path("/tmp/submission_to_validate.csv")
    submission_file.save(submission_path)

    try:
        is_valid, result = run_validation(submission_path, competition_id)
    except Exception as e:
        # Server error
        return jsonify({"error": "An unexpected error occurred.", "details": str(e)}), 500

    return jsonify({"result": result, "is_valid": is_valid})

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "running"}), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
