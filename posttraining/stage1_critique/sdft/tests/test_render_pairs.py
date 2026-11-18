import copy
import json
import unittest
from render_sdft_pairs import render_pair


class EchoTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return json.dumps({'messages': messages, **kwargs}, sort_keys=True)


class RenderTests(unittest.TestCase):
    def test_passes_tools_and_history_without_mutation(self):
        messages = [{'role': 'system', 'content': 'rules'}, {'role': 'user', 'content': 'target'},
                    {'role': 'assistant', 'tool_calls': [{'id': 'a', 'function': {'name': 'check', 'arguments': '{}'}}]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': 'observed'}]
        tools = [{'type': 'function', 'function': {'name': 'check'}}]
        before = copy.deepcopy(messages)
        row = render_pair(messages, tools, 'PRIVATE_REFERENCE', EchoTokenizer())
        self.assertEqual(messages, before)
        self.assertNotIn('PRIVATE_REFERENCE', row['prompt'])
        self.assertIn('PRIVATE_REFERENCE', row['privileged_context'])
        a, b = json.loads(row['prompt']), json.loads(row['privileged_context'])
        self.assertEqual(a['tools'], tools)
        self.assertEqual(b['tools'], tools)
        self.assertEqual(a['messages'][2]['tool_calls'][0]['function']['arguments'], {})
        self.assertEqual(a['messages'][2:], b['messages'][2:])
        self.assertTrue(a['add_generation_prompt'])

    def test_rejects_invalid_json_tool_arguments(self):
        for value in ['{broken', '{"x":1,"x":2}', '{"x":NaN}', '[]', 'null']:
            messages = [{'role': 'user', 'content': 'target'},
                        {'role': 'assistant', 'tool_calls': [{'function': {'name': 'check', 'arguments': value}}]}]
            with self.assertRaises(ValueError):
                render_pair(messages, [{}], 'ref', EchoTokenizer())

    def test_rejects_flattened_non_native_tool_call_role(self):
        with self.assertRaises(ValueError):
            render_pair([{'role': 'tool_call', 'content': '{}'}], [{}], 'reference', EchoTokenizer())

    def test_rejects_missing_tools_or_reference(self):
        for tools, reference in [([], 'ref'), ([{}], '')]:
            with self.assertRaises(ValueError):
                render_pair([{'role': 'user', 'content': 'target'}], tools, reference, EchoTokenizer())


if __name__ == '__main__':
    unittest.main()
