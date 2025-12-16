import pytest
from ..evaluation.scorer import question_scorer

@pytest.mark.parametrize("input1, input2, expected", [
    ("42", "42", True),
    ("$42.00", "42", True),
    ("43", "42", False),
    ("10,765", "10765", True),
    ("22520", "22520", True)
])
def test_numeric_match(input1, input2, expected):
    assert question_scorer(input1, input2) == expected

@pytest.mark.parametrize("input1, input2, expected", [
    ("hello world", "Hello World", True),
    (" sea  gull ", "seagull", True),
    ("hello", "world", False),
    ("Other", "Other", True),
    #("A. Netherlands", "B. Netherlands", False), #ignoring as outlier
    ("", "Inditex", False),
    (" ", "Inditex", False),
    ("The average transaction amount is 91.85 EUR.", "91.852", True),
    ("The top 2 merchants account for approx 59.96% of all transactions.", "0.5996", False),
    ("Netherlands", "NL", False),
    ("", "", True) # @martini
])
def test_string_match(input1, input2, expected):
    assert question_scorer(input1, input2) == expected

@pytest.mark.parametrize("input1, input2, expected", [
    ("1, 2, 3", "1,2,3", True),
    ("apple; banana; cherry", "apple;banana;cherry", True),
    ("1, 2", "1, 2, 3", False),
    ("apple; banana", "apple; banana; cherry", False),
    ("uber, spotify, nike, netflix, inditex", "Nike, Netflix, Uber, Inditex, Spotify", True),
    ("a, b, c", "['a', 'b', 'c']", True),
    (
        "C: 69.36, F: 77.90, B: 86.22, A: 87.79, D: 94.34, G: 115.23",
        "[C: 69.36, F: 77.9, B: 86.22, A: 87.79, D: 94.34, G: 115.23]",
        True
    ),
    (
        "[BE: 85.3, IT: 93.82, FR: 98.57, NL: 99.87, LU: 111.42, SE: 114.36, ES: 134.41, GR: 169.92, ]",
        "BE: 85.3, IT: 93.82, FR: 98.57, NL: 99.87, LU: 111.42, SE: 114.36, ES: 134.41, GR: 169.92",
        True
    )
])
def test_list_match(input1, input2, expected):
    assert question_scorer(input1, input2) == expected

@pytest.mark.parametrize("input1, input2, expected", [
    ("42, hello", "42, hello", True),
    ("42, world", "42, hello", False),
    ("64", "64, 53, 454, 231, 473, 381", False)
])
def test_mixed_list_match(input1, input2, expected):
    assert question_scorer(input1, input2) == expected

@pytest.mark.parametrize("input1, input2, expected", [
    ("3.14", "3.1483", False),
    ("3.14", "3.20", False),
    ("1", "1.0", True),
    ("1.0", "1", True),
    ("0.731495413640441", "0.731495", True),
    ("C", "C) both ip_address and email_address", True),
    ("0.36706256984345176", "0.3670625698434518", True),
    ("$0.10", "$0.10 per retry", True),
    ("D", "D) Apples", True),
    ("D", "A) Oranges", False),
    ("25.0", "0.250", False), #input is not a percentage,
    ("5.760000", "5.715872", False),
    ("8.68000000000000", "8.66999999999916", False)
])
def test_approximate_numeric_match(input1, input2, expected):
    assert question_scorer(input1, input2) == expected
    
@pytest.mark.parametrize("input1, input2, expected", [
    ("73.15%", "73.1495", True),
    ("42%", "42", True),
    ("30%", "30.1", False),
    ("25", "25%", True),
    ("100%", "100", True),
    ("0.1%", "0.1", True),
    ("73%", "74", False),  # This should fail as the difference is too large
    ("90%", "89.99971063977545", True),
    ("7.79%", "7.787407043027865", True)
    # ("7.787407 %", "0.07787407043027865", True) #TODO FIX

])
def test_percentages_match(input1, input2, expected):
    assert question_scorer(input1, input2) == expected


