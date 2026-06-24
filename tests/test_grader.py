import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.grader import get_grade, summarize_grades

def test_grade_a():
    result = get_grade(90.0, 60.0)
    assert result == "Grade A"

def test_grade_b():
    result = get_grade(65.0, 40.0)
    assert result == "Grade B"

def test_grade_c():
    result = get_grade(45.0, 30.0)
    assert result == "Grade C"

def test_grade_d():
    result = get_grade(35.0, 15.0)
    assert result == "Grade D"

def test_grade_longest_dimensions():
    result = get_grade(35.0, 85.0)
    assert result == "Grade A"

# ------ TEST summary_grades() -------
def test_summarize_grades_count_correctly():
    detections = [
        {"grade": "Grade A"},
        {"grade": "Grade B"},
        {"grade": "Grade A"},
        {"grade": "Grade D"},
    ]
    result = summarize_grades(detections)

    assert result["Grade A"] == 2
    assert result["Grade B"] == 1
    assert result["Grade C"] == 0
    assert result["Grade D"] == 1

def test_summarize_grades_empty():
    result = summarize_grades([])
    assert result == {"Grade A": 0, "Grade B": 0, "Grade C": 0, "Grade D": 0,}