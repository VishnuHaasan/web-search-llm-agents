# 4get fixtures

Real responses from the local 4get (`http://localhost/api/v1`), saved once so tests run without the network.
Don't edit them by hand except where noted; re-capture with the same request if 4get's format changes.

| File | Request | Captured |
|---|---|---|
| `web_ddg.json` | `/web?...&scraper=ddg`, an asyncio query (exact text not recorded) | 2026-11-03 |
| `news_ddg.json` | `/news?...&scraper=ddg`, a python query (exact text not recorded) | 2026-11-03 |
| `web_answer_ddg.json` | `/web?s=python programming language&scraper=ddg` (answer box) | 2026-11-04 |
| `web_spelling_ddg.json` | `/web?s=pyhton asyncoi&scraper=ddg` (spelling `including`) | 2026-11-04 |
| `web_images_ddg.json` | `/web?s=python logo&scraper=ddg` (29-image box inside `/web`) | 2026-11-04 |
| `images_ddg.json` | `/images?s=python logo&scraper=ddg`, trimmed from 98 to 15 images | 2026-11-04 |
| `web_brave.json` | `/web?...&scraper=brave` while Brave was blocking | 2026-11-03 |
| `error_captcha.json` | hand-written, exact status text 4get sent for a ddg captcha | 2026-11-06 |
| `error_bad_npt.json` | hand-written, exact status text 4get sent for a bad `npt` | 2026-11-06 |

4get returns HTTP 200 for all of these, including the errors; the error is only in `status`.
