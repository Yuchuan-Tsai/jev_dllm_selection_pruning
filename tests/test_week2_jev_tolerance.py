"""Listwise choice tolerance for two-decimal rounding (Jev protocol v2); no network."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from jev_dllm.experiments import week2_jev
from jev_dllm.judges.listwise import parse_listwise_response


def request(ids):
    return {'model': 'jev-1.13.0', 'questions': {'best_answer': {'criteria': {cid: '' for cid in ids}}}}


def response(probabilities, choice, model='jev-1.13.0'):
    return {'model': model, 'answers': {'best_answer': {'type': 'choice', 'choice': choice, 'confidence': .23,
                                                         'probabilities': probabilities}},
            'usage': {'input_tokens': 900, 'output_tokens': 5}}


def spread(top, chosen, n=8):
    """Two-decimal probabilities: c0 holds the top value, c1 the chosen one, the rest share the remainder."""
    ids = [f'q/c{i:02}' for i in range(n)]
    rest = round((1 - top - chosen) / (n - 2), 2)
    probabilities = {ids[0]: top, ids[1]: chosen, **{cid: rest for cid in ids[2:]}}
    probabilities[ids[2]] = round(1 - top - chosen - rest * (n - 3), 2)
    return ids, probabilities


class ListwiseToleranceTests(unittest.TestCase):
    def test_one_rounding_step_below_the_top_is_accepted_and_flagged(self):
        # The three live rejections had these shapes: 0.27 vs 0.28, 0.14 vs 0.15, 0.40 vs 0.41.
        for top, chosen in ((.28, .27), (.15, .14), (.41, .40)):
            with self.subTest(top=top):
                ids, probabilities = spread(top, chosen)
                parsed = parse_listwise_response(response(probabilities, ids[1]), request(ids))
                self.assertEqual((parsed['candidate_id'], parsed['rounding_tie']), (ids[1], True))

    def test_consistent_choice_is_not_flagged(self):
        ids, probabilities = spread(.5, .2)
        parsed = parse_listwise_response(response(probabilities, ids[0]), request(ids))
        self.assertFalse(parsed['rounding_tie'])

    def test_larger_gaps_and_unrounded_probabilities_are_still_rejected(self):
        ids, probabilities = spread(.30, .28)
        with self.assertRaisesRegex(ValueError, 'beyond one two-decimal rounding step'):
            parse_listwise_response(response(probabilities, ids[1]), request(ids))
        ids = ['a', 'b']
        with self.assertRaisesRegex(ValueError, 'beyond one'):
            parse_listwise_response(response({'a': .5025, 'b': .4975}, 'b'), request(ids))

    def test_other_validation_failures_are_unchanged(self):
        ids, probabilities = spread(.28, .27)
        with self.assertRaisesRegex(ValueError, 'different model'):
            parse_listwise_response(response(probabilities, ids[1], model='jev-9.9.9'), request(ids))
        bad = response(probabilities, ids[1])
        bad['answers']['best_answer']['confidence'] = 1.5
        with self.assertRaisesRegex(ValueError, 'confidence'):
            parse_listwise_response(bad, request(ids))

    def test_protocol_version_and_tie_counting(self):
        self.assertEqual(week2_jev.VERSION, 'week2_jev_v2')
        records = [{'kind': 'listwise', 'rounding_tie': True, 'request_seconds': .2, 'usage': {'input_tokens': 1, 'output_tokens': 0}},
                   {'kind': 'listwise', 'rounding_tie': False, 'request_seconds': .2, 'usage': {'input_tokens': 1, 'output_tokens': 0}},
                   {'kind': 'pointwise', 'request_seconds': .1, 'usage': {'input_tokens': 1, 'output_tokens': 0}}]
        self.assertEqual(week2_jev.usage_summary(records)['listwise']['rounding_ties'], 1)


if __name__ == '__main__':
    unittest.main()
