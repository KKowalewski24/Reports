from typing import Callable, List

TILDE: str = "~"
UTF_8: str = "UTF-8"
INPUT_FILENAME: str = "input.txt"
OUTPUT_FILENAME: str = "output.txt"


def main() -> None:
    read_write(lambda words: insert_tilde(words))
    # read_write(lambda words: words)


def read_write(callback: Callable[[str], str]) -> None:
    words: str = ""
    with open(INPUT_FILENAME, encoding=UTF_8) as file:
        for line in file:
            words += line.strip().rstrip("\n") + " "
        words = words.strip()

    with open(OUTPUT_FILENAME, mode="w", encoding=UTF_8) as file:
        file.write(callback(words))


def insert_tilde(words: str) -> str:
    space_indexes: List[int] = [index for index, word in enumerate(words) if word.isspace()]
    words_list: List[str] = list(words)
    for space_index in space_indexes:
        if words[space_index - 2].isspace() or (space_index - 2) < 0:
            words_list[space_index] = TILDE

    return "".join(words_list)


if __name__ == "__main__":
    main()
