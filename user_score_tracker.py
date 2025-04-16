import json

def store_user_score(user_id, quiz_id, score, db_path="user_scores.json"):
    try:
        with open(db_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    if user_id not in data:
        data[user_id] = {}

    data[user_id][quiz_id] = score

    with open(db_path, 'w') as f:
        json.dump(data, f, indent=4)
