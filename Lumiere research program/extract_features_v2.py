import os
import time
import json
import pandas as pd
from groq import Groq

def find_input_file():
    candidate_paths = [
        "../data /clean_data_with_features.csv",
        "../data/clean_data_with_features.csv",
        "../data /clean_data.csv",
        "../data/clean_data.csv",
        "./data /clean_data_with_features.csv",
        "./data/clean_data_with_features.csv",
        "./clean_data_with_features.csv"
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None

def get_text_models(client):
    try:
        available_models = [m.id for m in client.models.list().data]
        preferred_order = [
            'qwen/qwen3.6-27b',
            'openai/gpt-oss-20b',
            'allam-2-7b'
        ]
        valid_models = [m for m in preferred_order if m in available_models]
        if not valid_models:
            valid_models = [
                m for m in available_models 
                if not any(x in m for x in ['whisper', 'compound', 'guard', 'orpheus'])
            ]
        print(f"Usable LLMs for processing: {valid_models}")
        return valid_models
    except Exception as e:
        print(f"Error fetching model list: {e}")
        return ['qwen/qwen3.6-27b']

def main():
    input_path = find_input_file()
    if not input_path:
        print("Error: Could not locate your input CSV file.")
        return

    output_dir = os.path.dirname(input_path)
    output_path = os.path.join(output_dir, "clean_data_with_features_v2.csv") if output_dir else "clean_data_with_features_v2.csv"

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY environment variable not set.")
        return

    client = Groq(api_key=api_key)
    usable_models = get_text_models(client)
    model_idx = 0

    df = pd.read_csv(input_path)
    print(f"Loaded base dataset from '{input_path}' with {len(df)} rows.")

    # Reset q1-q7 columns for the new questions
    for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']:
        df[q] = None

    # Only load previous progress if v2 file already exists (in case execution was interrupted)
    if os.path.exists(output_path):
        try:
            df_v2 = pd.read_csv(output_path)
            print(f"Resuming progress from existing '{output_path}'...")
            for q in ['q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7']:
                if q in df_v2.columns:
                    df[q] = df_v2[q]
        except Exception as e:
            print(f"Starting fresh for v2 output ({e}).")

    system_prompt = (
        "You are a financial analyst. Answer each question with exactly one of: "
        "'Yes', 'No', or 'Difficult to tell'. "
        "Return ONLY a JSON object with keys q1 through q7. Example: "
        '{"q1": "Yes", "q2": "No", "q3": "Yes", "q4": "No", "q5": "Difficult to tell", "q6": "Yes", "q7": "No"}'
    )

    questions = (
        "q1: Does the narrative provide concrete, numerical details about expected business profit or specific unit economics, rather than just a vague intention to buy and sell?\n"
        "q2: Does the loan description explicitly detail the borrower's previous business experience or track record in this specific industry?\n"
        "q3: Does the description rely heavily on the borrower's personal hardships, family size, or emotional struggles to justify the loan, rather than highlighting their business acumen?\n"
        "q4: Does the narrative focus more on the social impact of the loan (e.g., feeding children, community benefit) than on the actual mechanics of the business operations?\n"
        "q5: Does the borrower's intended activity heavily depend on agriculture, weather, livestock or any other factor that makes it highly exposed or susceptible to experience unpredictable natural shocks?\n"
        "q6: Is there explicit mention of the borrower holding a specific leadership role in a community group or having a long-standing, multi-year relationship with the field partner?\n"
        "q7: Does the narrative mention a specific, structured plan or alternative income source for how the loan will be repaid if the primary business fails?\n"
    )

    valid_responses = {'Yes', 'No', 'Difficult to tell'}

    def row_is_valid(row_data):
        for i in range(1, 8):
            val = str(row_data.get(f'q{i}', '')).strip()
            if pd.isna(row_data.get(f'q{i}')) or val not in valid_responses:
                return False
        return True

    rows_to_process = [idx for idx, row in df.iterrows() if not row_is_valid(row)]
    print(f"Found {len(rows_to_process)} row(s) requiring processing out of {len(df)} total rows.\n")

    for count, idx in enumerate(rows_to_process, start=1):
        row = df.loc[idx]
        desc = str(row.get('description', ''))
        user_content = f"{questions}\nLoan Description:\n{desc}"
        
        result = {}
        max_retries = len(usable_models) * 2

        for attempt in range(max_retries):
            current_model = usable_models[model_idx % len(usable_models)]
            try:
                response = client.chat.completions.create(
                    model=current_model,
                    max_tokens=1000,
                    temperature=0.0,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content}
                    ]
                )

                response_text = response.choices[0].message.content or ""

                if '</think>' in response_text:
                    response_text = response_text.split('</think>')[-1].strip()

                start = response_text.find('{')
                end = response_text.rfind('}')
                if start != -1 and end != -1:
                    json_str = response_text[start:end+1]
                else:
                    json_str = response_text.strip()

                result = json.loads(json_str)
                print(f"[{count}/{len(rows_to_process)}] Row {idx} Processed ({current_model}): {result}")
                break 

            except Exception as e:
                print(f"Error on row {idx} using {current_model}: {e}")
                model_idx += 1
                next_model = usable_models[model_idx % len(usable_models)]
                print(f"Switching model to '{next_model}'...")
                time.sleep(1.5)

        for i in range(1, 8):
            df.at[idx, f'q{i}'] = result.get(f'q{i}', 'Error')

        if count % 10 == 0:
            df.to_csv(output_path, index=False)

        time.sleep(1.0)

    df.to_csv(output_path, index=False)
    print(f"\nExecution Complete! Saved features to: {output_path}")

if __name__ == '__main__':
    main()