import io
import json
import os
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from kadai_physicalai.catalog import TASK_NAMES, load_catalog, select_task
from kadai_physicalai.openai_router import (
    OpenAIRouterError, api_key_from_environment_or_bashrc,
    parse_task_response, route,
)


def answer(task='turning_on_radio', reason='ラジオをオンにする'):
    return {
        'status': 'completed',
        'output': [{'type': 'message', 'content': [{
            'type': 'output_text',
            'text': json.dumps({'task_name': task, 'reason': reason}),
        }]}],
    }


class FakeHTTPResponse(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


class OpenAIRouterTests(unittest.TestCase):
    def test_key_from_environment(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': 'sk-only-for-test'}):
            with patch('subprocess.run') as run:
                self.assertEqual(api_key_from_environment_or_bashrc(), 'sk-only-for-test')
                run.assert_not_called()

    def test_key_bashrc_interactive_even_with_noninteractive_guard(self):
        with tempfile.TemporaryDirectory() as d:
            with open(os.path.join(d, '.bashrc'), 'w') as file:
                file.write('case $- in *i*) ;; *) return;; esac\nexport OPENAI_API_KEY=sk-from-test-bashrc\n')
            with patch.dict(os.environ, {'HOME': d, 'OPENAI_API_KEY': ''}):
                self.assertEqual(api_key_from_environment_or_bashrc(), 'sk-from-test-bashrc')

    def test_missing_key_fails_without_leaking(self):
        with patch.dict(os.environ, {'OPENAI_API_KEY': ''}):
            with patch('subprocess.run', return_value=subprocess.CompletedProcess([], 0, b'hi\x00\x00', b'')):
                with self.assertRaisesRegex(OpenAIRouterError, 'OPENAI_API_KEY'):
                    api_key_from_environment_or_bashrc()

    def test_openai_request_is_structured_and_uses_official_tasks(self):
        requests = []
        def fetch(req, timeout):
            requests.append(req)
            return FakeHTTPResponse(json.dumps(answer()).encode('utf-8'))
        with patch('kadai_physicalai.openai_router.api_key_from_environment_or_bashrc', return_value='sk-secret'):
            with patch('urllib.request.urlopen', side_effect=fetch):
                task, reason = route('ラジオをつけて', load_catalog())
        self.assertEqual(task, 'turning_on_radio')
        self.assertIn('ラジオ', reason)
        self.assertEqual(len(requests), 1)
        req = requests[0]
        self.assertEqual(req.full_url, 'https://api.openai.com/v1/responses')
        self.assertEqual(req.get_method(), 'POST')
        self.assertEqual(req.get_header('Authorization'), 'Bearer sk-secret')
        data = json.loads(req.data)
        self.assertFalse(data['store'])
        self.assertEqual(data['model'], 'gpt-4.1-mini')
        self.assertEqual(data['input'], 'ラジオをつけて')
        self.assertEqual(data['text']['format']['type'], 'json_schema')
        self.assertEqual(set(data['text']['format']['schema']['properties']['task_name']['enum']),
                         set(TASK_NAMES) | {'NONE'})

    def test_unknown_task_not_executed(self):
        with self.assertRaisesRegex(OpenAIRouterError, '一致する公式タスクがありません'):
            parse_task_response(answer('NONE', '該当しない'), load_catalog())
        with self.assertRaisesRegex(OpenAIRouterError, '未登録のタスクID'):
            parse_task_response(answer('delete_everything'), load_catalog())

    def test_bad_response_fails_closed(self):
        with self.assertRaises(OpenAIRouterError):
            parse_task_response({'status': 'incomplete', 'output': []}, load_catalog())
        with self.assertRaises(OpenAIRouterError):
            parse_task_response({'status': 'completed', 'output': []}, load_catalog())

    def test_error_does_not_leak_credentials(self):
        def fetch(req, timeout):
            raise HTTPError(req.full_url, 401, 'Unauthorized', {}, None)
        with patch('kadai_physicalai.openai_router.api_key_from_environment_or_bashrc', return_value='sk-secret'):
            with patch('urllib.request.urlopen', side_effect=fetch):
                with self.assertRaises(OpenAIRouterError) as ctx:
                    route('ラジオをつけて', load_catalog())
        self.assertIn('401', str(ctx.exception))
        self.assertNotIn('sk-secret', str(ctx.exception))

    def test_explicit_task_id_needs_no_api(self):
        with patch('kadai_physicalai.openai_router.route') as mock:
            self.assertEqual(select_task('turning_on_radio')[0], 'turning_on_radio')
            mock.assert_not_called()

    def test_japanese_uses_openai(self):
        with patch('kadai_physicalai.openai_router.route', return_value=('bringing_water', '水運搬')) as mock:
            result, why = select_task('冷蔵庫の水をリビングのテーブルに持ってきて')
            self.assertEqual(result, 'bringing_water')
            self.assertIn('OpenAI', why)
            mock.assert_called_once()

if __name__ == '__main__':
    unittest.main()
