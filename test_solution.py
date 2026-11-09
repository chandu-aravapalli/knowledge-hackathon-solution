"""Regression tests for the supplied hackathon snapshots."""
from pathlib import Path
import sys
import unittest

KIT = next((p for p in [Path.cwd(), *Path.cwd().parents] if (p / 'harness' / 'contract.py').exists()), None)
if KIT is None:
    raise SystemExit('Run tests from the root of the hackathon kit.')
sys.path.insert(0, str(KIT / 'harness'))
from run_eval import load_snapshots
from solution import build

class SolutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.kb = build()
        for snap in load_snapshots(KIT / 'snapshots'):
            cls.kb.ingest(snap)

    def query(self, question, date=None):
        return self.kb.answer(question, as_of=date).text

    def test_history(self):
        a = self.query('What is the 10-month jumbo certificate APY?', '2026-05-31')
        b = self.query('What is the 10-month jumbo certificate APY?')
        self.assertIn('3.90%', a)
        self.assertNotIn('4.00%', a)
        self.assertIn('4.00%', b)
        self.assertNotIn('3.90%', b)

    def test_auto_tier(self):
        self.assertIn('5.39%', self.query('What is the 60-month auto APR on a 2025 vehicle?'))
        self.assertIn('5.59%', self.query('What is the APR on an 84-month auto loan on a 2024 model?'))

    def test_balance_tier(self):
        a = self.query('What is Share Savings APY for $10,000?')
        self.assertIn('0.15%', a)
        self.assertNotIn('0.20%', a)
        self.assertIn('2.50%', self.query('What is 1st Rate Checking APY for $10,000?'))

    def test_rate_dimensions(self):
        a = self.query('What are the rates on a 30-year conventional mortgage?')
        self.assertIn('7.13%', a)
        self.assertIn('7.25%', a)
        self.assertIn('7.27%', a)
        self.assertIn('7.29%', a)

    def test_certificates(self):
        self.assertIn('3.40%', self.query('What was the 6-month regular certificate APY in May 2026?'))
        a = self.query('What is the 17-month special APY?')
        self.assertIn('4.10%', a)
        self.assertIn('4.20%', a)

    def test_unknown(self):
        for q in ('What is the rate on a 7/1 ARM?', 'What is the construction-loan rate?',
                  'What are the lobby hours of the Mullan Avenue branch?'):
            self.assertFalse(self.kb.answer(q).confident, q)
        self.assertFalse(self.kb.answer('Anything?', as_of='2025-01-01').confident)

    def test_repeatable(self):
        q = 'What is the interest rate on Bright Savings?'
        self.assertEqual(self.query(q), self.query(q))

    def test_inline_dates(self):
        self.assertIn('3.90%', self.query('What was the 10-month jumbo certificate APY as of May 2026?'))
        for question in ('What was the 10-month jumbo certificate APY as of May 1, 2026?',
                         'What was the 10-month jumbo certificate APY in April 2026?'):
            self.assertFalse(self.kb.answer(question).confident)

    def test_written_loan_terms(self):
        answer = self.query('What APR for a five-year auto loan on a 2025 model?')
        self.assertIn('5.39%', answer)
        self.assertNotIn('5.19%', answer)
        self.assertNotIn('5.49%', answer)
        self.assertNotIn('5.59%', answer)

    def test_mortgage_loan_amount_boundaries(self):
        answer = self.kb.answer('What is the rate on a $10,000 30-year conventional home purchase loan?')
        self.assertTrue(answer.confident)
        self.assertIn('7.13%', answer.text)
        self.assertNotIn('7.25%', answer.text)
        for amount in ('$1,000,000', '$100,000,000'):
            self.assertFalse(self.kb.answer(
                f'What is the rate on a {amount} 30-year conventional home purchase loan?'
            ).confident)

    def test_no_hallucinated_products(self):
        for question in ('What is the APY for Cosmic Diamond Savings account?',
                         'Does Dupaco offer insurance for teleportation failures?'):
            self.assertFalse(self.kb.answer(question).confident, question)

    def test_conflicts(self):
        values = {tuple(c.values) for c in self.kb.conflicts()}
        self.assertIn(('5.34%', '5.39%'), values)
        self.assertIn(('3.90%', '4.00%'), values)

if __name__ == '__main__':
    unittest.main(verbosity=2)
