from bs4 import BeautifulSoup
import requests
from dataclasses import dataclass
import json
from pathlib import Path

BASE_URL = "https://prase.cz/kalva/putnam/"
YEAR_URL = "https://prase.cz/kalva/putnam/putn{index}.html"
BASE_IDX = 38


@dataclass
class Problem:
    title: str
    problem: str
    problem_text: str
    solution_text: str

    def __str__(self):
        return f"{self.title}\n{self.problem}\n{self.problem_text}\n{self.solution_text}\n"

    def to_json(self) -> dict:
        return {
            "title": self.title,
            "problem": self.problem,
            "problem_text": self.problem_text,
            "solution_text": self.solution_text
        }


def get_problem(url: str):
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    title = soup.find("h3").text
    problem = soup.find("b").text.strip().removeprefix("Problem ").lower()

    p_tags = soup.find_all("p")

    before_solution = []
    after_solution = []
    is_before = True
    for p_tag in p_tags:
        tags = p_tag.contents[:-1]
        if any([tag.name == "b" and "Solution" in tag.text for tag in tags]):
            is_before = False
        if any([tag.name == "img" and "../../line2.gif" in tag.get("src") for tag in tags]):
            break
        if is_before:
            before_solution.append(p_tag)
        else:
            after_solution.append(p_tag)

    problem_text = " ".join([content.text for p_tag in before_solution for content in p_tag.contents[:-1]])
    solution_text = " ".join([content.text for p_tag in after_solution for content in p_tag.contents[:-1]])

    return Problem(title, problem, problem_text, solution_text)


def get_year(year: str | int):
    url = YEAR_URL.format(index=year)
    page = requests.get(url)
    soup = BeautifulSoup(page.content, "html.parser")

    a_tags = soup.find_all("a")
    links = [link.get("href") for link in a_tags]
    problems = []
    for link in links:
        if link.startswith("psoln"):
            problems.append(get_problem(BASE_URL + link))

    return problems


def get_all():
    for year in range(BASE_IDX, 100):
        problems = get_year(year)
        for problem in problems:
            json_obj = problem.to_json()
            with open(f"putnam/putnam_19{year}_{problem.problem}.json", "w") as f:
                json.dump(json_obj, f, indent=4)
    for year in range(0, 10):
        problems = get_year("0" + str(year))
        for problem in problems:
            json_obj = problem.to_json()
            with open(f"putnam/putnam_200{year}_{problem.problem}.json", "w") as f:
                json.dump(json_obj, f, indent=4)

if __name__ == '__main__':
    Path("putnam").mkdir(parents=True, exist_ok=True)
    get_all()
