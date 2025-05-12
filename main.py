from smartquiz_generator import generate_questions, generate_answer
from user_score_tracker import store_user_score
from flashcard_generator import generate_flashcards_csv

def run():
    print("Welcome to SmartQuiz: AI Learning Assistant\n")
    print("Paste your course content below to auto-generate flashcards.\n")

    text = input("Paste your content here:\n\n")

    print("\nGenerating Questions...\n")
    questions = generate_questions(text)
    for i, q in enumerate(questions, 1):
        print(f"Q{i}: {q}")

    print("\nGenerating Answers & Saving Flashcards...\n")
    generate_flashcards_csv(questions, text, filename="anki_flashcards.csv")
    print("Flashcards saved to anki_flashcards.csv")

    print("\nStoring User Score (demo)...")
    store_user_score("student01", "demo_quiz", 0.88)
    print("Score saved to user_scores.json")

if __name__ == "__main__":
    run()
