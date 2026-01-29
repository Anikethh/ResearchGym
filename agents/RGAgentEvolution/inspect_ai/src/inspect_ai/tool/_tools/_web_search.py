import os
from typing import Literal, Protocol, runtime_checkable

import anyio
import httpx
from bs4 import BeautifulSoup, NavigableString
from tenacity import (
    retry,
    retry_if_exception,
    stop_after_attempt,
    stop_after_delay,
    wait_exponential_jitter,
)

from inspect_ai._util.error import PrerequisiteError
from inspect_ai._util.httpx import httpx_should_retry, log_httpx_retry_attempt
from inspect_ai.util._concurrency import concurrency

from .._tool import Tool, ToolResult, tool

DEFAULT_RELEVANCE_PROMPT = """I am trying to answer the following question and need to find the most relevant information on the web. Please let me know if the following content is relevant to the question or not. You should just respond with "yes" or "no".

Question: {question}
Page Content: {text}
"""


class SearchLink:
    def __init__(self, url: str, snippet: str, title: str | None = None, text: str | None = None) -> None:
        self.url = url
        self.snippet = snippet
        self.title = title
        self.text = text


@runtime_checkable
class SearchProvider(Protocol):
    async def __call__(self, query: str, start_idx: int) -> list[SearchLink]: ...


@tool
def web_search(
    provider: Literal["google", "exa"] = "google",
    num_results: int = 3,
    max_provider_calls: int = 3,
    max_connections: int = 10,
    model: str | None = None,
    relevance: Literal["model", "none"] = "model",
) -> Tool:
    """Web search tool.

    A tool that can be registered for use by models to search the web. Use
    the `use_tools()` solver to make the tool available (e.g. `use_tools(web_search())`))

    A web search is conducted using the specified provider, the results are parsed for relevance
    using the specified model, and the top 'num_results' relevant pages are returned.

    See further documentation at <https://inspect.aisi.org.uk/tools-standard.html#sec-web-search>.

    Args:
      provider: Search provider ("google" or "exa").
      num_results: Number of web search result pages to return to the model.
      max_provider_calls: Maximum number of search calls to make to the search provider.
      max_connections: Maximum number of concurrent connections to API
        endpoint of search provider.
      model: Model used to parse web pages for relevance (when relevance="model").
      relevance: Choose "model" to ask the model to judge page relevance, or
        "none" to skip LLM relevance and return provider snippets/content directly.

    Returns:
       A tool that can be registered for use by models to search the web.
    """
    # get search client (use a browsery UA + reasonable timeouts to reduce 403s)
    client = httpx.AsyncClient(
        headers={
            "User-Agent": (
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/123.0.0.0 Safari/537.36"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        },
        timeout=httpx.Timeout(connect=10.0, read=15.0, write=15.0, pool=None),
        follow_redirects=True,
    )

    if provider == "google":
        search_provider = google_search_provider(client)
    elif provider == "exa":
        search_provider = exa_search_provider(num_results)
    else:
        raise ValueError(
            f"Provider {provider} not supported. Supported providers are 'google' and 'exa'."
        )

    # For Exa, skip model-based relevance by default (use provider snippets)
    if provider == "exa":
        relevance = "none"

    # resolve provider (only google for now)
    async def execute(query: str) -> ToolResult:
        """
        Use the web_search tool to perform keyword searches of the web.

        Args:
            query (str): Search query.
        """
        # limit number of concurrent searches
        page_contents: list[str] = []
        urls: list[str] = []
        # snippets from successfully fetched pages
        snippets: list[str] = []
        # snippets from all links (for fallback)
        all_snippets: list[str] = []
        # structured full-content results for Exa
        exa_structured: list[str] = []
        search_calls = 0

        # Paginate through search results until we have enough
        while len(page_contents) < num_results and search_calls < max_provider_calls:
            async with concurrency(f"{provider}_web_search", max_connections):
                links = await search_provider(query, start_idx=search_calls * 10)

            if relevance == "none":
                # Do not fetch pages or call LLM; prefer Exa full text when available
                for lk in links:
                    if provider == "exa" and getattr(lk, "text", None):
                        title = getattr(lk, "title", "") or ""
                        url = getattr(lk, "url", "") or ""
                        text = getattr(lk, "text", "") or ""
                        exa_structured.append(
                            f"Title: {title}\nURL: {url}\nText:\n{text}"
                        )
                        if len(exa_structured) >= num_results:
                            break
                    elif lk.snippet:
                        all_snippets.append(lk.snippet)
                        if len(all_snippets) >= num_results:
                            break
                if (provider == "exa" and len(exa_structured) >= num_results) or (
                    len(all_snippets) >= num_results
                ):
                    break
            else:
                async with anyio.create_task_group() as tg:

                    async def process_link(link: SearchLink) -> None:
                        try:
                            page = await page_if_relevant(link.url, query, model, client)
                            if page:
                                page_contents.append(page)
                                urls.append(link.url)
                                snippets.append(link.snippet)
                        # exceptions fetching pages are very common!
                        except Exception:
                            pass

                    for lk in links:
                        # collect snippet for possible fallback even if fetch fails
                        try:
                            if lk.snippet:
                                all_snippets.append(lk.snippet)
                        except Exception:
                            pass
                        tg.start_soon(process_link, lk)

            search_calls += 1

        all_page_contents = "\n\n".join(page_contents)
        if provider == "exa" and exa_structured:
            response = (
                "Here are web search results (full content):\n\n"
                + "\n\n---\n\n".join(exa_structured[: max(1, num_results)])
            )
        elif all_page_contents == "" or relevance == "none":
            # Use provider snippets (either as fallback or by configuration)
            fallback = "\n\n".join(all_snippets[: max(1, num_results)])
            if fallback.strip():
                response = (
                    "Here are web search snippets (summaries). They may be useful later! "
                    + fallback
                )
            else:
                response = (
                    "I'm sorry, I couldn't find any relevant information on the web."
                )
        else:
            response = (
                "Here are your web search results. Please read them carefully as they may be useful later! "
                + all_page_contents
            )

        return response

    return execute


async def page_if_relevant(
    link: str, query: str, relevance_model: str | None, client: httpx.AsyncClient
) -> str | None:
    """
    Use parser model to determine if a web page contents is relevant to a query.

    Args:
        link (str): Web page link.
        query (str): Search query.
        relevance_model (Model): Model used to parse web pages for relevance.
        client: (httpx.Client): HTTP client to use to fetch the page

    Returns:
        str: Web page contents if relevant, else None.
    """
    # resolve model
    from inspect_ai.model._model import get_model

    model = get_model(relevance_model)

    # retrieve document
    try:
        response = await client.get(link)
        response.raise_for_status()
    except httpx.HTTPError:
        # Defer to snippet fallback upstream
        return None

    # parse it
    encoding_scheme = response.encoding or "utf-8"
    soup = BeautifulSoup(response.content.decode(encoding_scheme), "html.parser")

    main_content = soup.find("main") or soup.find("body") or soup
    if not isinstance(main_content, NavigableString):
        paragraphs = main_content.find_all("p")
        full_text = ""
        for p in paragraphs:
            full_text += p.get_text(strip=True, separator=" ")
            if len(full_text.split()) > 2000:
                break
    else:
        full_text = " ".join(
            main_content.get_text(strip=True, separator=" ").split()[:2000]
        )

    is_relevant = (
        await model.generate(
            DEFAULT_RELEVANCE_PROMPT.format(question=query, text=full_text)
        )
    ).message.text

    if "yes" in is_relevant.lower():
        return full_text
    else:
        return None


def google_search_provider(client: httpx.AsyncClient) -> SearchProvider:
    google_api_key = os.environ.get("GOOGLE_CSE_API_KEY", None)
    google_cse_id = os.environ.get("GOOGLE_CSE_ID", None)
    if not google_api_key or not google_cse_id:
        raise PrerequisiteError(
            "GOOGLE_CSE_ID and/or GOOGLE_CSE_API_KEY not set in the environment. Please ensure these variables are defined to use Google Custom Search with the web_search tool.\n\nLearn more about the Google web search provider at https://inspect.aisi.org.uk/tools.html#google-provider"
        )

    async def search(query: str, start_idx: int) -> list[SearchLink]:
        # List of allowed parameters can be found https://developers.google.com/custom-search/v1/reference/rest/v1/cse/list
        search_params = {
            "q": query,
            "key": google_api_key,
            "cx": google_cse_id,
            "start": start_idx,
        }
        search_url = "https://www.googleapis.com/customsearch/v1?" + "&".join(
            [f"{key}={value}" for key, value in search_params.items()]
        )

        # retry up to 5 times over a period of up to 1 minute
        @retry(
            wait=wait_exponential_jitter(),
            stop=stop_after_attempt(5) | stop_after_delay(60),
            retry=retry_if_exception(httpx_should_retry),
            before_sleep=log_httpx_retry_attempt(search_url),
        )
        async def execute_search() -> httpx.Response:
            return await client.get(search_url)

        result = await execute_search()
        data = result.json()

        if "items" in data:
            return [SearchLink(item["link"], item["snippet"]) for item in data["items"]]
        else:
            return []

    return search


def exa_search_provider(num_results: int) -> SearchProvider:
    exa_api_key = os.environ.get("EXA_API_KEY", None)
    if not exa_api_key:
        raise PrerequisiteError(
            "EXA_API_KEY not set in the environment. Please set EXA_API_KEY to use the Exa provider."
        )

    try:
        from exa_py import Exa  # type: ignore
    except Exception as exc:
        raise PrerequisiteError(
            "The 'exa_py' package is not installed. Install it to use the Exa provider (e.g. `pip install exa_py`)."
        ) from exc

    exa = Exa(api_key=exa_api_key)
    seen_urls: set[str] = set()

    async def search(query: str, start_idx: int) -> list[SearchLink]:
        def _search_and_contents():
            # Keep arguments conservative for SDK compatibility.
            return exa.search_and_contents(
                query=query,
                type="auto",
                text=True,
                # highlights=True,
                # category=["research paper", "github"],
                end_published_date="2024-12-31T00:00:00.000Z",
                num_results=num_results,
            )

        try:
            data = await anyio.to_thread.run_sync(_search_and_contents)
        except Exception:
            return []

        # Normalize results
        results = None
        if isinstance(data, dict):
            results = data.get("results") or data.get("documents") or data.get("items")
        if results is None and hasattr(data, "results"):
            results = getattr(data, "results")
        if results is None and isinstance(data, list):
            results = data
        if not isinstance(results, list):
            results = []

        links: list[SearchLink] = []
        for item in results:
            try:
                url = (
                    (item.get("url") if isinstance(item, dict) else getattr(item, "url", None))
                    or (item.get("web_url") if isinstance(item, dict) else getattr(item, "web_url", None))
                )
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)

                # Extract title and text; build a short snippet as fallback
                raw_h = item.get("highlights") if isinstance(item, dict) else getattr(item, "highlights", None)
                title = (
                    (item.get("title") if isinstance(item, dict) else getattr(item, "title", None))
                    or ""
                )
                text = (
                    (item.get("text") if isinstance(item, dict) else getattr(item, "text", None))
                    or ""
                )
                snippet = ""
                if isinstance(raw_h, list) and raw_h:
                    snippet = " ".join(str(raw_h[0]).split())[:300]
                elif isinstance(raw_h, str) and raw_h:
                    snippet = " ".join(raw_h.split())[:300]
                elif text:
                    snippet = " ".join(str(text).split())[:300]
                else:
                    snippet = title

                links.append(SearchLink(url=url, snippet=snippet, title=title, text=text))
            except Exception:
                continue

        return links

    return search
