"""Wire tap contract: the exact JSON body POSTed to /chat/completions and the
parsed response are appended, per call, to a JSONL file. Recorder snapshots
taken before dispatch are NOT sufficient for training admission; neither is this log alone.
"""
import json
import hashlib
from http.server import BaseHTTPRequestHandler, HTTPServer
import os
import tempfile
import threading
import unittest
from unittest import mock

from core import llm_client
from core.llm_client import LLMClient


def _fake_stream_reader(data):
    def _read(resp, model=None):
        return data
    return _read


class WireTapTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "wire_log.jsonl")
        self.client = LLMClient(base_url="http://x/v1", model="m", max_tokens=32,
                                fallback_enabled=False, wire_log_path=self.path)
        self.ok = {"id": "r1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"},
                                             "finish_reason": "stop"}], "usage": {"prompt_tokens": 5}}

    def tearDown(self):
        self.tmp.cleanup()

    def _post(self, messages, tools=None, data=None):
        posted = []
        fake_resp = mock.MagicMock()
        fake_resp.raise_for_status.return_value = None
        def post(url, json=None, **kw):
            posted.append(json)
            return fake_resp
        with mock.patch.object(self.client.session, "post", side_effect=post), \
             mock.patch.object(llm_client, "read_chat_completions_stream", _fake_stream_reader(data or self.ok)):
            out = self.client.chat(messages, tools=tools)
        return posted, out

    def test_logged_request_equals_submitted_payload(self):
        msgs = [{"role": "system", "content": "s"},
                {"role": "user", "content": "u"},
                {"role": "assistant", "content": None, "tool_calls": [{"id": "c1", "type": "function",
                    "function": {"name": "f", "arguments": "{\"a\": 1}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}]
        tools = [{"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"a": {"type": "integer"}}}}}]
        posted, out = self._post(msgs, tools)
        rows = [json.loads(l) for l in open(self.path)]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["request"], posted[0])
        self.assertEqual(json.dumps(rows[0]["request"], sort_keys=True), json.dumps(posted[0], sort_keys=True))
        self.assertEqual(rows[0]["response"], self.ok)
        self.assertEqual(rows[0]["url"], "http://x/v1/chat/completions")
        self.assertEqual(rows[0]["outcome"], "ok")
        self.assertIn("request_sha256", rows[0])
        self.assertEqual(rows[0]["attempt"], 1)
        self.assertIs(rows[0]["prepared_request_verified"], False)
        self.assertIsNone(rows[0]["raw_request_body_utf8"])
        self.assertIsNone(rows[0]["status_code"])

    def test_prepared_request_bytes_and_response_identity(self):
        bodies = []
        def post(url, json=None, **kwargs):
            response = llm_client.requests.Response()
            response.status_code = 200
            response.request = llm_client.requests.Request("POST", url, json=json).prepare()
            bodies.append(response.request.body)
            return response
        data = {**self.ok, "model": "actual-model"}
        with mock.patch.object(self.client.session, "post", side_effect=post), \
             mock.patch.object(llm_client, "read_chat_completions_stream", return_value=data):
            self.client.chat([{"role": "user", "content": "你好"}])
        row = json.loads(open(self.path).read())
        self.assertIs(row["prepared_request_verified"], True)
        self.assertEqual(row["raw_request_body_utf8"].encode(), bodies[0])
        self.assertEqual(json.loads(bodies[0]), row["request"])
        self.assertEqual(row["raw_request_body_sha256"], hashlib.sha256(bodies[0]).hexdigest())
        self.assertEqual(row["status_code"], 200)
        self.assertEqual(row["response"]["id"], "r1")
        self.assertEqual(row["response"]["model"], "actual-model")
        self.assertTrue(row["call_id"])

    def test_loopback_http_captures_actual_bytes(self):
        bodies = []
        chunk = {"id": "local-response", "model": "local-model", "choices": [
            {"index": 0, "delta": {"content": "local ok"}, "finish_reason": "stop"}
        ]}
        stream = ("data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n").encode()

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                bodies.append(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Content-Length", str(len(stream)))
                self.end_headers()
                self.wfile.write(stream)

            def log_message(self, *args):
                pass

        try:
            server = HTTPServer(("127.0.0.1", 0), Handler)
        except PermissionError as error:
            self.skipTest(f"Loopback sockets unavailable: {error}")
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        self.client.base_url = f"http://127.0.0.1:{server.server_port}/v1"
        self.client.session.trust_env = False
        try:
            self.client.chat([{"role": "user", "content": "local only"}])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)
        row = json.loads(open(self.path).read())
        self.assertIs(row["prepared_request_verified"], True)
        self.assertEqual(row["raw_request_body_utf8"].encode(), bodies[0])
        self.assertEqual(json.loads(bodies[0]), row["request"])
        self.assertEqual(row["raw_request_body_sha256"], hashlib.sha256(bodies[0]).hexdigest())
        self.assertEqual(row["response"]["id"], "local-response")
        self.assertEqual(row["response"]["model"], "local-model")
        self.assertEqual(row["status_code"], 200)

    def test_generic_parse_failure_is_recorded_and_raised(self):
        with mock.patch.object(self.client.session, "post", return_value=mock.MagicMock()), \
             mock.patch.object(llm_client, "read_chat_completions_stream", side_effect=ValueError("bad stream")):
            with self.assertRaisesRegex(ValueError, "bad stream"):
                self.client.chat([{"role": "user", "content": "u"}])
        row = json.loads(open(self.path).read())
        self.assertEqual(row["outcome"], "error")
        self.assertIsNone(row["response"])

    def test_recovery_errors_preserve_original_count_and_call_ids(self):
        self.client.max_retries = 1
        empty = {"choices": []}
        with mock.patch.object(llm_client.requests.Session, "post", return_value=mock.MagicMock()), \
             mock.patch.object(llm_client, "read_chat_completions_stream", side_effect=[empty, ValueError("broken recovery"), self.ok]), \
             mock.patch.object(llm_client.time, "sleep"):
            self.client.chat([{"role": "user", "content": "u"}, {"role": "tool", "tool_call_id": "ghost", "content": "x"}])
        rows = [json.loads(line) for line in open(self.path)]
        self.assertEqual([row["outcome"] for row in rows], ["empty", "recovery_error", "recovery_ok"])
        self.assertEqual([row["input_message_count"] for row in rows], [2, 2, 2])
        self.assertEqual(len({row["call_id"] for row in rows}), 3)
        self.assertEqual([row["attempt"] for row in rows], [1, 2, 3])

    def test_capture_write_failure_is_fatal_in_main_and_recovery(self):
        for in_recovery in (False, True):
            with self.subTest(in_recovery=in_recovery):
                self.client.max_retries = 1
                count = 0
                def read(response, model=None):
                    nonlocal count
                    count += 1
                    if in_recovery and count == 1:
                        return {"choices": []}
                    self.client.wire_log_path = self.tmp.name
                    return self.ok
                self.client.wire_log_path = self.path
                with mock.patch.object(llm_client.requests.Session, "post", return_value=mock.MagicMock()), \
                     mock.patch.object(llm_client, "read_chat_completions_stream", side_effect=read), \
                     mock.patch.object(llm_client.time, "sleep"):
                    with self.assertRaises(llm_client.CaptureIntegrityError):
                        self.client.chat([{"role": "user", "content": "u"}])

    def test_timeout_is_recorded(self):
        self.client.max_retries = 1
        with mock.patch.object(self.client.session, "post", side_effect=llm_client.requests.Timeout("timeout")), \
             mock.patch.object(llm_client.time, "sleep"):
            with self.assertRaises(llm_client.requests.Timeout):
                self.client.chat([{"role": "user", "content": "u"}])
        row = json.loads(open(self.path).read())
        self.assertEqual(row["outcome"], "error")
        self.assertTrue(row["call_id"])

    def test_overflow_is_logged_before_budget_mutation(self):
        from tests.test_llm_client_context_overflow import VLLM_MSG, _http_400
        self.client.max_tokens = 32768
        self.client.context_window = 65536
        error = _http_400(VLLM_MSG)
        with mock.patch.object(self.client.session, "post", side_effect=[error, mock.MagicMock()]), \
             mock.patch.object(self.client, "_exact_prompt_tokens", return_value=33381), \
             mock.patch.object(llm_client, "read_chat_completions_stream", return_value=self.ok):
            self.client.chat([{"role": "user", "content": "u"}])
        rows = [json.loads(line) for line in open(self.path)]
        self.assertEqual(rows[0]["request"]["max_tokens"], 32768)
        self.assertEqual(rows[1]["request"]["max_tokens"], 65536 - 33381 - 256)
        self.assertEqual(rows[0]["status_code"], 400)

    def test_orphan_tool_drop_is_visible_in_wire_not_hidden(self):
        # a tool result whose call is missing gets dropped before POST; the wire log must show the POSTED form
        msgs = [{"role": "user", "content": "u"}, {"role": "tool", "tool_call_id": "ghost", "content": "x"}]
        posted, _ = self._post(msgs)
        rows = [json.loads(l) for l in open(self.path)]
        self.assertEqual(rows[0]["request"]["messages"], posted[0]["messages"])
        self.assertEqual(rows[0]["input_message_count"], 2)
        self.assertEqual(len(rows[0]["request"]["messages"]), 1)

    def test_every_attempt_logged_including_failed(self):
        fake_ok = mock.MagicMock(); fake_ok.raise_for_status.return_value = None
        calls = {"n": 0}
        def post(url, json=None, **kw):
            calls["n"] += 1
            if calls["n"] == 1:
                raise llm_client.requests.ConnectionError("boom")
            return fake_ok
        with mock.patch.object(llm_client.requests.Session, "post", autospec=True, side_effect=lambda s, url, json=None, **kw: post(url, json=json, **kw)), \
             mock.patch.object(llm_client, "read_chat_completions_stream", _fake_stream_reader(self.ok)), \
             mock.patch.object(llm_client.time, "sleep"):
            self.client.chat([{"role": "user", "content": "u"}])
        rows = [json.loads(l) for l in open(self.path)]
        self.assertEqual([r["outcome"] for r in rows], ["error", "ok"])
        self.assertEqual([r["attempt"] for r in rows], [1, 2])
        self.assertIn("ConnectionError", rows[0]["error"])

    def test_no_path_means_no_file_and_no_behaviour_change(self):
        c = LLMClient(base_url="http://x/v1", model="m", fallback_enabled=False)
        self.assertIsNone(c.wire_log_path)
        fake = mock.MagicMock(); fake.raise_for_status.return_value = None
        with mock.patch.object(c.session, "post", return_value=fake), \
             mock.patch.object(llm_client, "read_chat_completions_stream", _fake_stream_reader(self.ok)):
            out = c.chat([{"role": "user", "content": "u"}])
        self.assertEqual(out["choices"][0]["message"]["content"], "hi")
        self.assertFalse(os.path.exists(self.path))


if __name__ == "__main__":
    unittest.main()
