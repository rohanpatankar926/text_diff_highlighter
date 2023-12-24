from module import SequenceMatcher

def compare_texts(actual_text, predicted_text):
    matcher = SequenceMatcher(None, actual_text.split(), predicted_text.split())

    differences = matcher.get_opcodes()

    for tag, i1, i2, j1, j2 in differences:
        if tag == 'replace':
            print(f"Replace from actual_text[{i1}:{i2}] -> predicted_text[{j1}:{j2}]:")
            print(f"Actual: {actual_text.split()[i1:i2]}")
            print(f"Predicted: {predicted_text.split()[j1:j2]}")
        elif tag == 'delete':
            print(f"Delete from actual_text[{i1}:{i2}]:")
            print(f"Actual: {actual_text.split()[i1:i2]}")
        elif tag == 'insert':
            print(f"Insert at predicted_text[{j1}:{j2}]:")
            print(f"Predicted: {predicted_text.split()[j1:j2]}")

actual_text = "This is the actual text for comparison of."
predicted_text = "this is the predicted text for comparison with some differences."

compare_texts(actual_text, predicted_text)
