import glob
import os
import pathlib
import shutil
import subprocess
import sys
import webbrowser
from typing import List

"""
* Building project `python main.py -r`
* Opening pdf file `python main.py -o`
* Cleaning project `python main.py -c`

* Building options 
    Check types - `python main.py -t`
    
"""

# VAR ------------------------------------------------------------------------ #
PDF_LATEX = "pdflatex"

TEX_FILENAME = "ADZ_Task5_Karwowski_Kowalewski"
PDF = ".pdf"
TEX = ".tex"

SOURCE_DIRECTORY = "src"
DIRECTORIES = ["auxil", "out"]
FILE_EXTENSIONS = [
    "*.aux", "*.bcf", "*.locode", "*.lof", "*.log",
    "*.lot", "*.lot", "*.pdf", "*.out", "*.xml", "*.gz", "*.toc"
]


# MAIN ----------------------------------------------------------------------- #
def main() -> None:
    if len(sys.argv) == 2:
        if sys.argv[1] == "clean" or sys.argv[1] == "-c":
            clean_project()
        elif sys.argv[1] == "run" or sys.argv[1] == "-r":
            build_project()
        elif sys.argv[1] == "open" or sys.argv[1] == "-o":
            open_generated_pdf()

    display_finish()


# DEF ------------------------------------------------------------------------ #
def clean_project() -> None:
    go_to_parent_directory(get_current_directory())

    try:
        remove_directories(DIRECTORIES)
    except FileNotFoundError:
        print("Some directories may not be found!")

    go_to_child_directory(SOURCE_DIRECTORY)

    try:
        for it in FILE_EXTENSIONS:
            remove_files(glob.glob(it))
    except FileNotFoundError:
        print("Some files may not be found!")


def open_generated_pdf() -> None:
    go_to_parent_directory(get_current_directory())
    go_to_child_directory(DIRECTORIES[1])
    webbrowser.open(TEX_FILENAME + PDF)


def build_project() -> None:
    go_to_parent_directory(get_current_directory())
    go_to_child_directory(SOURCE_DIRECTORY)

    generate_pdf()
    generate_pdf()


def generate_pdf() -> None:
    subprocess.call(
        [PDF_LATEX,
         "-file-line-error",
         "-interaction=nonstopmode",
         "-synctex=1",
         "-output-format=pdf",
         "-output-directory=../" + DIRECTORIES[1],
         "-aux-directory=../" + DIRECTORIES[0],
         TEX_FILENAME + TEX],
        shell=True
    )


# UTIL ----------------------------------------------------------------------- #
def remove_files(filenames: List[str]) -> None:
    for it in filenames:
        os.remove(it)


def remove_directories(directory_names: List[str]) -> None:
    for it in directory_names:
        shutil.rmtree(it)


def get_current_directory():
    return pathlib.Path(os.getcwd())


def go_to_parent_directory(directory) -> None:
    os.chdir(directory.parent)


def go_to_child_directory(directory: str) -> None:
    os.chdir(directory)


def check_types() -> None:
    subprocess.call(["mypy", "."])


def display_finish() -> None:
    print("------------------------------------------------------------------------")
    print("FINISHED")
    print("------------------------------------------------------------------------")


# __MAIN__ ------------------------------------------------------------------- #
if __name__ == "__main__":
    if len(sys.argv) == 2 and (sys.argv[1] == "typing" or sys.argv[1] == "-t"):
        check_types()
    else:
        main()
