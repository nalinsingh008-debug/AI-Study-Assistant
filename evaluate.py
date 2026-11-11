
import csv
from datetime import datetime
from capstone import connect_spur, find_related_answer

test_questions = [
    "What is Python?",
    "What is artificial intelligence?",
    "Explain a variable in Python.",
    "What is a Python list?",
    "What is a tuple?",
    "Explain a dictionary in Python.",
    "What is a loop?",
    "What is a function?",
    "What is a string?",
    "What is an integer?",
    "What is a float?",
    "What is a Boolean value?",
    "What is machine learning?",
    "Explain data science.",
    "What is an algorithm?",
    "What is a Python class?",
    "What is an object?",
    "What is an if statement?",
    "Why do programmers use comments?",
    "What is the difference between for and while loops?",
    "How can Python help beginners learn AI?",
    "Give an example of a Python variable."
]

def main():
    print("AI STUDY ASSISTANT - EVALUATION")
    print("Total test questions:", len(test_questions))

    try:
        client, model = connect_spur()
        print("SPUR connected successfully!")
        print("Selected model:", model)
    except Exception as error:
        print("SPUR connection failed:", error)
        return

    results = []

    for number, question in enumerate(test_questions, start=1):
        print(f"\nTest {number}/{len(test_questions)}")
        print("Question:", question)

        related_answer = find_related_answer(question)

        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a beginner-friendly study assistant. "
                            "Give accurate, simple answers with examples "
                            "when useful."
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"Question: {question}\n"
                            f"Related knowledge: {related_answer}\n"
                            "Answer in simple language."
                        )
                    }
                ]
            )

            answer = response.choices[0].message.content or ""
            status = "Success"
            print("AI:", answer)

        except Exception as error:
            answer = str(error)
            status = "Failed"
            print("Error:", error)

        results.append({
            "test_number": number,
            "question": question,
            "related_knowledge": related_answer,
            "ai_answer": answer,
            "status": status,
            "accuracy_score_0_to_2": "",
            "relevance_score_0_to_2": "",
            "clarity_score_0_to_2": ""
        })

    filename = "evaluation_results.csv"

    with open(filename, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    successful = sum(
        1 for result in results if result["status"] == "Success"
    )

    print("\n========== EVALUATION SUMMARY ==========")
    print("Total questions:", len(results))
    print("Successful responses:", successful)
    print("Failed responses:", len(results) - successful)
    print("Results saved in:", filename)
    print("Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("========================================")

if __name__ == "__main__":
    main()
