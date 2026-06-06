import csv
import json

from core.intent import detect_intent


def main() -> None:
    results = []

    with open("tests.csv", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            question = row["问题"].strip()
            need_search = row["应搜索"].strip() == "是"
            intent = detect_intent(question)
            actual_search = bool(intent["need_search"])

            results.append(
                {
                    "question": question,
                    "need_search": need_search,
                    "actual_search": actual_search,
                    "correct": need_search == actual_search,
                }
            )

    for result in results:
        print(json.dumps(result, ensure_ascii=False))

    correct_count = sum(1 for result in results if result["correct"])
    total_count = len(results)
    accuracy = correct_count / total_count if total_count else 0

    print()
    print("Intent Accuracy")
    print(f"{correct_count}/{total_count}")
    print(f"{accuracy:.0%}")


if __name__ == "__main__":
    main()
