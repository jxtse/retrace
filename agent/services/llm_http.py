"""HTTP session shared by the harness LLM transports."""
import requests


def _reject_redirect(response, **kwargs):
    # A response hook runs before requests processes Location or reads the body.
    if 300 <= response.status_code < 400:
        response.close()
        raise requests.HTTPError(
            f"LLM endpoint returned HTTP {response.status_code}; configure the final endpoint URL.",
            response=response,
        )


class LLMHTTPSession(requests.Session):
    def __init__(self):
        super().__init__()
        self.hooks["response"].append(_reject_redirect)

    def request(self, method, url, **kwargs):
        kwargs["allow_redirects"] = False
        return super().request(method, url, **kwargs)
