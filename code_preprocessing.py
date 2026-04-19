import json

def clean_comment_fast(code):
    res = []
    state = 0

    for c in code:
        if state == 0 and c == '/':
            state = 1
        elif state == 1 and c == '*':
            state = 2
        elif state == 1 and c == '/':
            state = 4
        elif state == 1:
            res.append(c)
            state = 0
        elif state == 2 and c == '*':
            state = 3
        elif state == 2:
            continue
        elif state == 3 and c == '/':
            state = 0
        elif state == 3:
            state = 2
        elif state == 4 and c == '\\':
            state = 9
        elif state == 9 and c == '\\':
            state = 9
        elif state == 9:
            state = 4
        elif state == 4 and c == '\n':
            state = 0
        elif state == 0 and c == '\'':
            state = 5
        elif state == 5 and c == '\\':
            state = 6
        elif state == 6:
            state = 5
        elif state == 5 and c == '\'':
            state = 0
        elif state == 0 and c == '\"':
            state = 7
        elif state == 7 and c == '\\':
            state = 8
        elif state == 8:
            state = 7
        elif state == 7 and c == '\"':
            state = 0

        if (state == 0 and c != '/') or state in {5,6,7,8}:
            res.append(c)

    return ''.join(res)


def process_json_safe(input_path, output_path, max_len=20000):
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    new_data = []

    for item in data:
        try:
            code = item.get("function", "")

            if not isinstance(code, str) or not code:
                continue

            if len(code) > max_len:
                continue

            item["function"] = clean_comment_fast(code)
            new_data.append(item)

        except Exception:
            continue

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=False)


process_json_safe("data.json", "cleaned.json")