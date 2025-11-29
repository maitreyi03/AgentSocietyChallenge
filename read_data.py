import json

def read_young_adults_data():
    """
    Reads the Goodreads young adults data and saves a slimmed-down version
    with only: isbn, book_id, title, genres.
    """
    input_path = "/Users/MaitreyiPareek/Desktop/AgentSocietyChallenge/data/goodreads_books_young_adult.json"
    output_path = "/Users/MaitreyiPareek/Desktop/AgentSocietyChallenge/data/genre_list.txt"
    
    slim_books = []

    with open(input_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            book_data = json.loads(line)

            isbn = book_data.get("isbn", "") or ""
            book_id = book_data.get("book_id", "") or ""
            title = book_data.get("title", "") or ""

            popular_shelves = book_data.get("popular_shelves", []) or []
            # Take all shelf names as "genres" (unique, sorted)
            genres = sorted({shelf.get("name", "") for shelf in popular_shelves if shelf.get("name")})

            slim_books.append({
                "isbn": isbn,
                "book_id": book_id,
                "title": title,
                "genres": genres,
            })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(slim_books, f, ensure_ascii=False, indent=2)

    return slim_books


read_young_adults_data()