
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

print("================================")
print("       AI STUDY ASSISTANT")
print("================================")

# Study knowledge database
questions = [
    "What is AI?",
    "What is Python?",
    "What is Machine Learning?",
    "What is Deep Learning?",
    "What is Data Science?",
    "What is NLP?",
    "What is a chatbot?",
    "What is an algorithm?",
    "What is a variable?",
    "What is a function?",
    "What is a dataset?",
    "What is Generative AI?",
    "What is automation?",
    "What is a neural network?",
    "What is computer vision?",
    "What is cloud computing?",
    "What is data analysis?",
    "What is a Python list?",
    "What is a Python loop?",
    "What is a dictionary?"
]

answers = [
    "AI means Artificial Intelligence. It helps machines perform intelligent tasks.",
    "Python is a popular programming language used in AI and software development.",
    "Machine Learning allows computers to learn patterns from data.",
    "Deep Learning uses multi-layered neural networks.",
    "Data Science involves collecting and analysing data.",
    "NLP helps computers understand human language.",
    "A chatbot is a program that communicates with users.",
    "An algorithm is a step-by-step method to solve a problem.",
    "A variable stores information in a program.",
    "A function is a reusable block of code.",
    "A dataset is a collection of related information.",
    "Generative AI creates new content like text, images and code.",
    "Automation uses technology to perform tasks automatically.",
    "A neural network is an AI model inspired by the human brain.",
    "Computer vision helps computers understand images and videos.",
    "Cloud computing provides computing services through the internet.",
    "Data analysis means examining data to find useful information.",
    "A Python list stores multiple items in an ordered collection.",
    "A loop repeats a block of code.",
    "A dictionary stores data in key-value pairs."
]

# AI Technique 1: TF-IDF
vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# AI Technique 2: Cosine Similarity
def get_answer(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(
        user_vector, question_vectors
    )

    best_match = similarity.argmax()
    confidence = similarity[0][best_match]

    if confidence < 0.15:
        return "Sorry, I don't know that answer. Try another question."

    return answers[best_match]



import os
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Knowledge base: 20 study questions
knowledge_base = [
    ("what is python", "Python is a popular programming language."),
    ("what is ai", "AI means Artificial Intelligence. It helps computers perform intelligent tasks."),
    ("what is a variable", "A variable stores information in a program."),
    ("what is a list", "A list stores multiple items in Python."),
    ("what is a tuple", "A tuple stores multiple items and cannot be changed."),
    ("what is a dictionary", "A dictionary stores information in key-value pairs."),
    ("what is a loop", "A loop repeats a block of code."),
    ("what is a function", "A function is reusable code that performs a task."),
    ("what is a string", "A string is text written inside quotes."),
    ("what is an integer", "An integer is a whole number."),
    ("what is a float", "A float is a number with a decimal point."),
    ("what is a boolean", "A Boolean value is either True or False."),
    ("what is machine learning", "Machine learning helps computers learn patterns from data."),
    ("what is data science", "Data science uses data to find useful information."),
    ("what is an algorithm", "An algorithm is a sequence of steps to solve a problem."),
    ("what is a class", "A class is a blueprint for creating objects in Python."),
    ("what is an object", "An object is an instance of a class."),
    ("what is a conditional statement", "A conditional statement runs code based on a condition."),
    ("what is a comment", "A comment explains code and is ignored by Python."),
    ("what is a loop in python", "Python uses for and while loops to repeat code.")
]

# 2. Prepare TF-IDF search
questions = [item[0] for item in knowledge_base]
answers = [item[1] for item in knowledge_base]

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)


def find_related_answer(user_question):
    user_vector = vectorizer.transform([user_question])
    scores = cosine_similarity(user_vector, question_vectors)
    best_index = scores.argmax()
    best_score = scores[0][best_index]

    if best_score < 0.15:
        return "No closely matching question found."

    return answers[best_index]


# 3. Connect to SPUR AI
def connect_spur():
    api_key = os.getenv("SPUR_API_KEY")

    if not api_key:
        raise ValueError("SPUR_API_KEY not found in environment.")

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
        raise ValueError("SPUR Chat Mini model not found.")

    return client, selected_model


# 4. Main chatbot
def main():
    print("===== AI STUDY ASSISTANT =====")

    try:
        client, model = connect_spur()
        print("SPUR connected successfully!")
        print("Selected model:", model)

        print("\nAsk me a study question.")
        print("Type 'exit' to stop.\n")

        while True:
            question = input("You: ").strip()

            if question.lower() == "exit":
                print("Goodbye! Keep learning.")
                break

            if not question:
                continue

            related_answer = find_related_answer(question)

            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a beginner-friendly study assistant. "
                                "Give accurate, simple explanations. "
                                "Use the provided knowledge base when relevant."
                            )
                        },
                        {
                            "role": "user",
                            "content": (
                                f"Question: {question}\n"
                                f"Related knowledge: {related_answer}\n"
                                "Explain the answer simply."
                            )
                        }
                    ]
                )

                print("\nAI:", response.choices[0].message.content)
                print()

            except Exception as error:
                print("\nSPUR request error:", error)

    except Exception as error:
        print("\nConnection error:", error)


if __name__ == "__main__":
    main()
