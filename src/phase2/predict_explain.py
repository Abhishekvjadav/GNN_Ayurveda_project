
import sys
import os

# Make phase2 imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from explain_prediction_v3 import explain


if __name__ == "__main__":

    if len(sys.argv) != 3:
        print()
        print("Usage:")
        print("python predict_explain.py <compound_id> <target_id>")
        print()
        print("Example:")
        print(
            "python predict_explain.py "
            "COMPOUND_806ebd096539 "
            "DISEASE_5e294e090c89"
        )
        sys.exit(1)

    compound_id = sys.argv[1]
    target_id = sys.argv[2]

    explain(compound_id, target_id)
