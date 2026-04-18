import os
from utils.parser import extract_text
from utils.preprocessor import preprocess_text


def load_resumes_for_role(target_role):
    roles = ["accountant", "chef", "information_technology"]

    texts = []
    labels = []

    for role in roles:
        folder = f"roles/{role}/resumes"

        for file in os.listdir(folder):
            file_path = os.path.join(folder, file)

            try:
                text = extract_text(file_path)
                clean_text = preprocess_text(text)

                texts.append(clean_text)

                # Binary labeling
                if role == target_role:
                    labels.append(1)
                else:
                    labels.append(0)

            except Exception as e:
                print(f"Error processing {file}: {e}")

    return texts, labels