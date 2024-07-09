import os
import json
from openai import OpenAI
from sentence_transformers import SentenceTransformer, util

class LLMProcessor:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)

    def ask_user_for_questions(self, max_questions=3):
        questions = []
        print(f"Please enter up to {max_questions} questions:")
        for i in range(max_questions):
            question = input(f"Enter question {i + 1}: ").strip()
            if question:
                questions.append(question)
            if len(questions) == max_questions:
                break
        print("--==+ Processing Responses +==--")
        return questions

    def ask_questions(self, questions, output_file='responses.json'):
        responses = []

        if os.path.exists(output_file):
            with open(output_file, 'r') as json_file:
                responses = json.load(json_file)

        for question in questions:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Let's pretend that I am a resume coach and career advisor. You are my client. I will ask you a question and you are to provide me a response relevant to that question. You are to be as detailed as possible while also being brief. You may be creative."},
                    {"role": "user", "content": f"{question}"}
                ]
            )

            response = completion.choices[0].message.content
            responses.append({"user1": question, "user2": response})

        with open(output_file, 'w') as json_file:
            json.dump(responses, json_file, indent=4)

        print(f"Responses saved to {output_file}")

    def process_with_llm(self, user1, user2):
        messages = [
            {"role": "system", "content": '''
                You are only to respond in json format.
                You will be given a transcript between two users. You are to identify the index of the question from the approved list and fill in the parameters respectively. Keep the parameters concise and general, leaving out unnecessary details.

                For example:
                {
                    "user1": "Please introduce yourself",
                    "user2":  "My name is Alice and I have a bachelors degree."
                }

                Response:
                {
                    "Education": ["Bachelors"], 
                    "Experience": [], 
                    "Skills": [], 
                    "Strengths": [], 
                    "Goals": [], 
                    "Values": []
                }
            '''},
            {"role": "user", "content": f'''
                {{
                    "user1": "{user1}",
                    "user2": "{user2}"
                }}
            '''}
        ]

        completion = self.client.chat.completions.create(
            model="gpt-4o",
            messages=messages
        )

        return completion.choices[0].message.content

    def merge_json(self, existing_data, new_data):
        for label, new_content in new_data.items():
            if label in existing_data:
                if isinstance(existing_data[label], list):
                    if isinstance(new_content, list):
                        existing_data[label].extend(item for item in new_content if item not in existing_data[label])
                    else:
                        if new_content not in existing_data[label]:
                            existing_data[label].append(new_content)
                else:
                    if existing_data[label] != new_content:
                        existing_data[label] = [existing_data[label], new_content] if isinstance(new_content, str) else [existing_data[label]] + new_content
            else:
                existing_data[label] = new_content

    def parse_json_response(self, response):
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            print("Error: The response is not valid JSON.")
            return None

    def process_responses(self, input_file='responses.json', output_file='processed_responses.json'):
        if os.path.exists(input_file):
            with open(input_file, 'r') as json_file:
                questions_responses = json.load(json_file)
        else:
            print("No responses.json file found.")
            questions_responses = []

        processed_responses = []
        existing_data = {}

        for index, item in enumerate(questions_responses):
            question = item["user1"]
            response = item["user2"]
            processed_response = self.process_with_llm(question, response)

            new_data = self.parse_json_response(processed_response)

            if new_data:
                self.merge_json(existing_data, new_data)
            else:
                print("Failed to merge data due to invalid JSON response.")
            
            processed_responses.append({
                "question": question,
                "response": response,
                "processed_response": processed_response
            })
        
        with open(output_file, 'w') as json_file:
            json.dump(processed_responses, json_file, indent=4)

        print(f"Processed responses saved to {output_file}")
        print("Final Merged Data:", json.dumps(existing_data, indent=4))
        return existing_data

    def rerank(self, data):
        # Desired job title
        print("Entire a goal or career choice that will rerank the list of labels?")
        desired_job = str(input())

        # Load pre-trained Sentence-BERT model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Encode the desired job title
        job_embedding = model.encode(desired_job, convert_to_tensor=True)

        # Function to rank items based on relevance
        def rank_items_by_relevance(data, job_embedding):
            ranked_data = {}
            for category, items in data.items():
                item_embeddings = model.encode(items, convert_to_tensor=True)
                scores = util.pytorch_cos_sim(job_embedding, item_embeddings)[0]
                ranked_items = [item for _, item in sorted(zip(scores, items), reverse=True)]
                ranked_data[category] = ranked_items
            return ranked_data

        # Rank items in each category
        ranked_data = rank_items_by_relevance(data, job_embedding)

        # Print ranked data
        print(json.dumps(ranked_data, indent=2))

    def run(self):
        questions = self.ask_user_for_questions()
        self.ask_questions(questions)
        data = self.process_responses()
        self.rerank(data)
