"""Network-free regression: effort must reach the HTTP send boundary for Qwen3.8."""
import unittest
from unittest.mock import patch
from pathlib import Path
from core.config import load_config
from core.llm_client import LLMClient, _build_chat_request
from services.llm_models import to_chat_completions

ROOT = Path(__file__).resolve().parents[1]

class WireTests(unittest.TestCase):
    def request(self, model='qwen3.8-27b-student', effort='low'):
        return _build_chat_request(model=model,
            messages=[{'role': 'user', 'content': 'Use the real tools.'}],
            tools=None, temperature=0.7, max_tokens=32768, reasoning_effort=effort)

    def test_all_official_efforts_survive_projection(self):
        for effort in ('low', 'medium', 'xhigh'):
            with self.subTest(effort=effort):
                wire = to_chat_completions(self.request(effort=effort)).model_dump(exclude_none=True)
                self.assertEqual(wire.get('reasoning_effort'), effort)
                self.assertEqual(wire['max_tokens'], 32768)

    def test_unverified_model_wire_stays_unchanged(self):
        wire = to_chat_completions(self.request(model='gpt-4o')).model_dump(exclude_none=True)
        self.assertNotIn('reasoning_effort', wire)

    def test_deployed_config_reaches_real_send_boundary(self):
        cfg = load_config(str(ROOT/'configs/student_27b_vllm.yaml'), thinking='fast')['llm']
        self.assertEqual((cfg['reasoning_effort'], cfg['max_tokens']), ('low', 32768))
        req = self.request(model=cfg['model'], effort=cfg['reasoning_effort'])
        client = LLMClient(base_url=cfg['base_url'], model=cfg['model'],
            reasoning_effort=cfg['reasoning_effort'], max_tokens=cfg['max_tokens'], fallback_enabled=False)
        with patch.object(client.session, 'post') as post, patch(
            'core.llm_client.read_chat_completions_stream',
            return_value={'choices':[{'message':{'content':'unit-test fixture, not model output'},'finish_reason':'stop'}]}):
            client._post_chat_completions(req, [{'role':'user','content':'Use the real tools.'}])
        wire = post.call_args.kwargs['json']
        self.assertEqual(wire.get('reasoning_effort'), 'low')
        self.assertEqual(wire['max_tokens'], 32768)
        self.assertTrue(wire['stream'])
        print('HTTP_SEND_BOUNDARY_PASS', {k:wire[k] for k in ('model','reasoning_effort','max_tokens')})

if __name__ == '__main__':
    unittest.main(verbosity=2)
