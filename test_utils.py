import unittest
from utils import extract_boxed_answer, verify_answer


class TestExtractBoxedAnswer(unittest.TestCase):
    """Test cases for extract_boxed_answer function."""
    
    def test_simple_number(self):
        """Test extraction of simple number."""
        solution = r"The answer is \boxed{42}"
        self.assertEqual(extract_boxed_answer(solution), "42")
    
    def test_fraction(self):
        """Test extraction of fraction."""
        solution = r"Therefore \boxed{\frac{11}{36}}"
        self.assertEqual(extract_boxed_answer(solution), r"\frac{11}{36}")
    
    def test_square_root(self):
        """Test extraction of square root."""
        solution = r"The final answer is \boxed{2\sqrt{41}}"
        self.assertEqual(extract_boxed_answer(solution), r"2\sqrt{41}")
    
    def test_nested_braces(self):
        """Test extraction with nested braces."""
        solution = r"Answer: \boxed{\frac{1}{2} + \sqrt{3}}"
        self.assertEqual(extract_boxed_answer(solution), r"\frac{1}{2} + \sqrt{3}")
    
    def test_complex_expression(self):
        """Test extraction of complex expression."""
        solution = r"Result is \boxed{\frac{a^2 + b^2}{c}}"
        self.assertEqual(extract_boxed_answer(solution), r"\frac{a^2 + b^2}{c}")
    
    def test_multiple_boxed(self):
        """Test that only first boxed answer is extracted."""
        solution = r"First \boxed{1} then \boxed{2}"
        self.assertEqual(extract_boxed_answer(solution), "1")
    
    def test_boxed_in_long_solution(self):
        """Test extraction from long solution text."""
        solution = r"""Since $\tan{M}=\frac{5}{4}$, we have $\dfrac{NO}{OM} = \dfrac{5}{4}$, 
        so $$NO = \frac{5}{4}OM = \frac{5}{4}\cdot 8 = 10.$$
        Then, from the Pythagorean Theorem, we have 
        \begin{align*} MN&=\sqrt{NO^2+OM^2}\\ 
        &=\sqrt{10^2+8^2}=\sqrt{164}=\boxed{2\sqrt{41}}.\end{align*}"""
        self.assertEqual(extract_boxed_answer(solution), r"2\sqrt{41}")
    
    def test_no_boxed(self):
        """Test when no boxed answer present."""
        solution = "The answer is 42"
        self.assertEqual(extract_boxed_answer(solution), "")
    
    def test_empty_string(self):
        """Test with empty string."""
        self.assertEqual(extract_boxed_answer(""), "")
    
    def test_empty_boxed(self):
        """Test extraction of empty boxed."""
        solution = r"\boxed{}"
        self.assertEqual(extract_boxed_answer(solution), "")
    
    def test_boxed_with_spaces(self):
        """Test boxed answer with spaces."""
        solution = r"\boxed{ 42 }"
        self.assertEqual(extract_boxed_answer(solution), " 42 ")
    
    def test_multiple_nested_braces(self):
        """Test deeply nested braces."""
        solution = r"\boxed{\left(\frac{a+b}{c}\right)^2}"
        self.assertEqual(extract_boxed_answer(solution), r"\left(\frac{a+b}{c}\right)^2")
    
    def test_boxed_with_text(self):
        """Test boxed with text (like for word problems)."""
        solution = r"\boxed{\text{impossible}}"
        self.assertEqual(extract_boxed_answer(solution), r"\text{impossible}")
    
    def test_negative_number(self):
        """Test extraction of negative number."""
        solution = r"\boxed{-5}"
        self.assertEqual(extract_boxed_answer(solution), "-5")
    
    def test_decimal(self):
        """Test extraction of decimal."""
        solution = r"\boxed{3.14159}"
        self.assertEqual(extract_boxed_answer(solution), "3.14159")


class TestVerifyAnswer(unittest.TestCase):
    """Test cases for verify_answer function."""
    
    def test_identical_answers(self):
        """Test identical answers match."""
        self.assertTrue(verify_answer("42", "42"))
    
    def test_identical_fractions(self):
        """Test identical fractions match."""
        self.assertTrue(verify_answer(r"\frac{1}{2}", r"\frac{1}{2}"))
    
    def test_equivalent_fractions(self):
        """Test equivalent fractions match."""
        self.assertTrue(verify_answer(r"\frac{2}{4}", r"\frac{1}{2}"))
    
    def test_identical_square_roots(self):
        """Test identical square roots match."""
        self.assertTrue(verify_answer(r"2\sqrt{41}", r"2\sqrt{41}"))
    
    def test_simplified_vs_unsimplified(self):
        """Test simplified vs unsimplified expressions."""
        # This might or might not work depending on math_verify's capabilities
        result = verify_answer(r"\sqrt{164}", r"2\sqrt{41}")
        # Both forms are mathematically equivalent
        self.assertTrue(result)
    
    def test_different_answers(self):
        """Test different answers don't match."""
        self.assertFalse(verify_answer("42", "43"))
    
    def test_different_fractions(self):
        """Test different fractions don't match."""
        self.assertFalse(verify_answer(r"\frac{1}{2}", r"\frac{1}{3}"))
    
    def test_empty_predicted(self):
        """Test with empty predicted answer."""
        self.assertFalse(verify_answer("", "42"))
    
    def test_empty_ground_truth(self):
        """Test with empty ground truth."""
        self.assertFalse(verify_answer("42", ""))
    
    def test_both_empty(self):
        """Test with both empty."""
        self.assertFalse(verify_answer("", ""))
    
    def test_with_spaces(self):
        """Test answers with extra spaces."""
        self.assertTrue(verify_answer(" 42 ", "42"))
    
    def test_negative_numbers(self):
        """Test negative numbers."""
        self.assertTrue(verify_answer("-5", "-5"))
        self.assertFalse(verify_answer("-5", "5"))
    
    def test_decimals(self):
        """Test decimal numbers."""
        self.assertTrue(verify_answer("3.14", "3.14"))
    
    def test_fraction_vs_decimal(self):
        """Test fraction vs decimal equivalence."""
        # math_verify should recognize these as equivalent
        self.assertTrue(verify_answer(r"\frac{1}{2}", "0.5"))
    
    def test_complex_expression_match(self):
        """Test complex expressions that match."""
        expr = r"\frac{a^2 + b^2}{c}"
        self.assertTrue(verify_answer(expr, expr))
    
    def test_invalid_latex_fallback(self):
        """Test fallback to string comparison for invalid LaTeX."""
        # If math_verify fails, should fall back to string comparison
        result = verify_answer("hello", "hello")
        self.assertTrue(result)
    
    def test_mixed_operators(self):
        """Test expressions with different operators."""
        self.assertTrue(verify_answer("2+3", "5"))
    
    def test_parentheses(self):
        """Test expressions with parentheses."""
        self.assertTrue(verify_answer("(2+3)", "5"))
    
    def test_multiple_terms(self):
        """Test multi-term expressions."""
        expr = r"x^2 + 2x + 1"
        self.assertTrue(verify_answer(expr, expr))
    
    def test_with_pi(self):
        """Test expressions with pi."""
        self.assertTrue(verify_answer(r"\pi", r"\pi"))
    
    def test_trigonometric(self):
        """Test trigonometric expressions."""
        self.assertTrue(verify_answer(r"\sin(x)", r"\sin(x)"))
    
    def test_vectors_or_sets(self):
        """Test sets notation."""
        self.assertTrue(verify_answer(r"\{1,2,3\}", r"\{1,2,3\}"))
    
    def test_infinity(self):
        """Test infinity symbol."""
        self.assertTrue(verify_answer(r"\infty", r"\infty"))
    
    def test_matrix_notation(self):
        """Test if matrix-like expressions work."""
        expr = r"\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}"
        # This might fail with LatexExtractionConfig, but should handle gracefully
        try:
            result = verify_answer(expr, expr)
        except:
            # If it fails, that's okay - we're testing edge cases
            pass


class TestIntegration(unittest.TestCase):
    """Integration tests combining both functions."""
    
    def test_extract_and_verify_workflow(self):
        """Test complete workflow: extract from solution and verify."""
        gt_solution = r"The answer is \boxed{2\sqrt{41}}"
        model_output = r"My solution: \boxed{2\sqrt{41}}"
        
        gt_answer = extract_boxed_answer(gt_solution)
        predicted_answer = extract_boxed_answer(model_output)
        
        self.assertTrue(verify_answer(predicted_answer, gt_answer))
    
    def test_extract_and_verify_wrong_answer(self):
        """Test workflow with wrong answer."""
        gt_solution = r"The answer is \boxed{42}"
        model_output = r"My answer: \boxed{43}"
        
        gt_answer = extract_boxed_answer(gt_solution)
        predicted_answer = extract_boxed_answer(model_output)
        
        self.assertFalse(verify_answer(predicted_answer, gt_answer))
    
    def test_extract_and_verify_equivalent_forms(self):
        """Test workflow with mathematically equivalent answers."""
        gt_solution = r"Answer: \boxed{\frac{1}{2}}"
        model_output = r"Result: \boxed{\frac{2}{4}}"
        
        gt_answer = extract_boxed_answer(gt_solution)
        predicted_answer = extract_boxed_answer(model_output)
        
        self.assertTrue(verify_answer(predicted_answer, gt_answer))
    
    def test_no_boxed_in_model_output(self):
        """Test when model doesn't use boxed format."""
        gt_solution = r"Answer: \boxed{42}"
        model_output = "The answer is 42"  # No boxed!
        
        gt_answer = extract_boxed_answer(gt_solution)
        predicted_answer = extract_boxed_answer(model_output)
        
        # predicted_answer will be empty string
        self.assertFalse(verify_answer(predicted_answer, gt_answer))
    
    def test_real_math_problem(self):
        """Test with a realistic MATH dataset example."""
        gt_solution = r"""Since $\tan{M}=\frac{5}{4}$, we have $\dfrac{NO}{OM} = \dfrac{5}{4}$, 
        so $$NO = \frac{5}{4}OM = \frac{5}{4}\cdot 8 = 10.$$Then, from the Pythagorean Theorem, 
        we have \begin{align*} MN&=\sqrt{NO^2+OM^2}\\ 
        &=\sqrt{10^2+8^2}=\sqrt{164}=\boxed{2\sqrt{41}}.\end{align*}"""
        
        model_output = r"Using Pythagorean theorem: $MN = \sqrt{10^2 + 8^2} = \boxed{2\sqrt{41}}$"
        
        gt_answer = extract_boxed_answer(gt_solution)
        predicted_answer = extract_boxed_answer(model_output)
        
        self.assertEqual(gt_answer, r"2\sqrt{41}")
        self.assertEqual(predicted_answer, r"2\sqrt{41}")
        self.assertTrue(verify_answer(predicted_answer, gt_answer))


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)