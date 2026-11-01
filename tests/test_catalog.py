import unittest
from kadai_physicalai.catalog import ALIASES, TASK_NAMES, TaskSelectionError, load_catalog, select_task, suggestions
from kadai_physicalai.runner import make_command

class CatalogTests(unittest.TestCase):
    def test_exactly_100(self):
        self.assertEqual(len(TASK_NAMES), 100)
        self.assertEqual(len(set(TASK_NAMES)), 100)

    def test_explicit_task(self):
        self.assertEqual(select_task('turning_on_radio')[0], 'turning_on_radio')

    def test_japanese(self):
        from unittest.mock import patch
        with patch('kadai_physicalai.openai_router.route', return_value=('bringing_water','水の運搬')):
            self.assertEqual(select_task('冷蔵庫の水をリビングのテーブルに持ってきて')[0], 'bringing_water')

    def test_aliases_all_valid(self):
        self.assertTrue(all(t in TASK_NAMES for t in ALIASES.values()))

    def test_no_green_cup_hallucination(self):
        from unittest.mock import patch
        from kadai_physicalai.openai_router import OpenAIRouterError
        with patch('kadai_physicalai.openai_router.route', side_effect=OpenAIRouterError('一致する公式タスクがありません')):
            with self.assertRaises(TaskSelectionError):
                select_task('キッチンに行って緑のコップを取って出発地点に戻って')

    def test_empty_fails(self):
        with self.assertRaises(TaskSelectionError):
            select_task('')

    def test_unknown_fails(self):
        from unittest.mock import patch
        from kadai_physicalai.openai_router import OpenAIRouterError
        with patch('kadai_physicalai.openai_router.route', side_effect=OpenAIRouterError('一致する公式タスクがありません')):
            with self.assertRaises(TaskSelectionError):
                select_task('月へ飛んでいって')

    def test_catalog_subset(self):
        self.assertEqual(set(load_catalog()), set(TASK_NAMES))

    def test_suggestion(self):
        self.assertIn('bringing_water', suggestions('bringing'))

    def test_valid_command(self):
        cmd = make_command('turning_on_radio', instance=0, headless=True)
        self.assertEqual(cmd[-3:], ['turning_on_radio', '0', '1'])

    def test_invalid_task(self):
        with self.assertRaises(ValueError):
            make_command('curl evil')

    def test_bad_instance(self):
        with self.assertRaises(ValueError):
            make_command('turning_on_radio', instance=20)

if __name__ == '__main__': unittest.main()
