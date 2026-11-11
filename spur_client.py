
import os
from openai import OpenAI

def main():
    print("===== AI STUDY ASSISTANT =====")

    api_key = os.getenv("SPUR_API_KEY")

    if not api_key:
        print("ERROR: SPUR_API_KEY not found.")
        return

    try:
        client = OpenAI(
            api_key=api_key,
            base_url="https://ai.spuric.com/v1"
        )

        models = client.models.list()
        selected_model = None

        for model in models.data:
            name = model.id.lower().replace("-", " ").replace("_", " ")
            if "chat" in name and "mini" in name:
                selected_model = model.id
                break

        if not selected_model:
            print("Chat Mini model not found.")
            return

        print("Model:", selected_model)
        print("\nYour AI Study Assistant is ready!")
        print("Type 'exit' to close the assistant.\n")

        while True:
            question = input("You: ").strip()

            if question.lower() == "exit":
                print("Goodbye! Keep learning.")
                break

            if not question:
                continue

            response = client.chat.completions.create(
                model=selected_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful study assistant. Explain answers in simple language."
                    },
                    {
                        "role": "user",
                        "content": question
                    }
                ]
            )

            answer = response.choices[0].message.content
            print("\nAI:", answer or "No response received.", "\n")

    except Exception as error:
        print("ERROR:", error)


if __name__ == "__main__":
    main()







