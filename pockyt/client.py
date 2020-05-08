from __future__ import absolute_import, print_function, unicode_literals, with_statement

import copy
import logging
import sys
import time
from os.path import join

import pandas as pd
import parse

from .api import API
from .compat import prompt
from .wrapper import Browser, FileSystem, Network

# FIXME: Remove this before submitting pull request.
try:
    from util.caching import cache_today
    from util.shared import initialize_logging
    initialize_logging(logging.DEBUG)
except:
    pass

class Client(object):
    """
    Pocket API Access Client
    """
    def __init__(self, credentials, args):
        self._args = args
        self._credentials = credentials
        self._api_endpoint = ""
        self._payload = {}
        self._response = None

        self._format_spec = ""
        self._unformat_spec = None
        self._output = []
        self._input = []

    def _api_request(self, payload=None, endpoint=None):
        if payload is None:
            payload = self._payload
        if endpoint is None:
            endpoint = self._api_endpoint

        # add API access credentials
        payload.update(self._credentials)

        # batch if necessary
        def __batch_payload(n):
            if "actions" not in payload:
                yield payload

            else:
                for ndx in range(0, len(payload["actions"]), n):
                    batch = copy.copy(payload)
                    batch["actions"] = payload["actions"][ndx:ndx+n]
                    yield batch

        # access API
        responses = []
        for batch in __batch_payload(100):
            logging.debug("Executing network request: %.1000s", batch)
            responses.append(Network.post_request(endpoint, batch))

        # FIXME: This is a big hack.
        self._response = responses[-1]
        return responses

    def _output_to_file(self):
        file_path = FileSystem.resolve_path(self._args.output)
        content = ''.join(
            map(lambda info: self._format_spec.format(**info), self._output))
        FileSystem.write_to_file(file_path, content)

    def _print_to_console(self, info):
        line = self._format_spec.format(**info)
        try:
            print(line, end="")
        except UnicodeEncodeError:
            print(line.encode(API.ENCODING), end="")

    def _open_in_browser(self, info):
        time.sleep(1)
        Browser.open_new_tab(info["link"])

    def _save_to_archive(self, info):
        archive_path = FileSystem.resolve_path(self._args.archive)
        FileSystem.ensure_dir((archive_path))
        title = FileSystem.get_safe_name(info['title'])
        filename = '{0} - {1}.html'.format(info['id'], title)
        filepath = join(archive_path, filename)
        html = Network.get_html(info['link'])
        FileSystem.write_to_file(filepath, html)

    def _get_console_input(self):
        print("Enter data: {0}".format(self._args.format.strip()))
        try:
            while True:
                data = prompt().strip()
                if data:
                    info = self._unformat_spec.parse(data)
                    self._input.append(info)
                else:
                    raise EOFError
        except EOFError:
            pass

    def _get_redirect_input(self):
        for line in sys.stdin.readlines():
            data = line.strip()
            if data:
                info = self._unformat_spec.parse(data)
                self._input.append(info)
            else:
                continue

    def _get_file_input(self):
        with open(self._args.input, "r") as f:
            for line in f.readlines():
                data = line.strip()
                if data:
                    info = self._unformat_spec.parse(data)
                    self._input.append(info)
                else:
                    continue

    def _get_args_input(self):
        info = self._unformat_spec.parse(self._args.input.strip())
        self._input.append(info)

    def _validate_format(self):
        # interpret escape sequences
        try:
            self._args.format = bytes(self._args.format,
                                      API.ENCODING).decode(API.DECODING)
        except TypeError:
            self._args.format = self._args.format.decode(API.DECODING)

        info = dict((key, None) for key in API.INFO_KEYS)

        try:
            self._args.format.format(**info)
        except KeyError:
            print("Invalid Format Specifier !")
            sys.exit(1)
        else:
            # NOTE: Addresses excerpts with newlines.
            self._format_spec = self._args.format.replace("{excerpt}", "{excerpt!r}") + "\n"
            self._unformat_spec = parse.compile(self._args.format)

    def _get(self):
        # create request payload
        payload = {
            "state": self._args.state,
            "sort": self._args.sort,
            "detailType": "complete",
        }

        if self._args.content != "all":
            payload["contentType"] = self._args.content

        if self._args.count != -1:
            payload["count"] = self._args.count

        if self._args.query:
            payload["search"] = self._args.query

        if self._args.tag == "-1":
            pass
        elif self._args.tag == "0":
            payload["tag"] = "_untagged_"
        else:
            payload["tag"] = self._args.tag

        if self._args.favorite != -1:
            payload["favorite"] = self._args.favorite

        if self._args.domain:
            payload["domain"] = self._args.domain

        self._payload = payload
        self._api_endpoint = API.RETRIEVE_URL

        self._api_request()

        items_df = self._parse_api_response(self._response)
        items_df["tags"] = items_df["tags"].apply(",".join)
        items_df.rename(columns={ "item_id": "id", "resolved_title": "title", "resolved_url": "link" }, inplace=True)
        items = tuple(items_df[["id", "title", "link", "excerpt", "tags"]].to_dict("records"))
        if len(items) == 0:
            print("No items found !")
            sys.exit(0)
        self._output = tuple(items)

    def _parse_api_response(self, responses):
        if not isinstance(responses, list):
            responses = [responses]

        parsed_df = pd.concat([pd.DataFrame(response.data.get("list", {})).T for response in responses])
        if "authors" in parsed_df:
            parsed_df["authors"] = parsed_df["authors"].apply(lambda x: tuple(y["name"] for y in x.values()) if pd.notnull(x) else pd.NA)
        if "tags" in parsed_df:
            parsed_df["tags"] = parsed_df["tags"].apply(lambda x: tuple(x.keys()) if pd.notnull(x) else ())  # NOTE: sklearn sortability

        # FIXME: Why do 14% of resolved_urls come back empty?
        return parsed_df

    def _clean_get(self, df):
        for col in ["given_title", "resolved_title", "resolved_url", "title", "link", "excerpt"]:
            if col not in df:
                continue
            df.loc[df[col].str.strip().str.len() == 0, col] = pd.NA
        for col in ["time_added", "time_updated", "time_read", "time_favorited"]:
            if col not in df:
                continue
            df[col] = pd.to_datetime(df[col], unit="s")
        for col in ["id", "item_id", "resolved_id", "sort_id", "time_to_read", "word_count", "listen_duration_estimate"]:
            if col not in df:
                continue
            df[col] = df[col].fillna(0).astype(int).astype("Int64")
            df.loc[df[col] == 0, col] = pd.NA
        if len({ "resolved_url", "given_url" } & set(df.columns)) == 2:
            df["resolved_url"] = df["resolved_url"].fillna(df["given_url"])
        if len({ "resolved_title", "given_title" } & set(df.columns)) == 2:
            df["resolved_title"] = df["resolved_title"].fillna(df["given_title"])
        return df

    def _put(self):
        actions = []
        for ix, info in enumerate(self._input):
            try:
                actions.append({
                    "action": "add",
                    "url": info["link"],
                    "title": info.named.get("title", ""),
                    "tags": info.named.get("tags", ""),
                })
            except:
                print("Skipping unparsed line {:}: {:}".format(ix+1, info))

        self._payload = { "actions": tuple(actions) }
        self._api_endpoint = API.MODIFY_URL

        self._api_request()

    def _modify(self):
        if self._args.delete:
            action = "delete"
        elif self._args.archive != -1:
            if self._args.archive == 1:
                action = "archive"
            else:
                action = "readd"
        elif self._args.favorite != -1:
            if self._args.favorite == 1:
                action = "favorite"
            else:
                action = "unfavorite"
        elif self._args.auto_tag:
            action = "tags_" + self._args.auto_tag
        else:
            action = ""

        if not action.startswith("tags_"):
            payload = {
                "actions":
                tuple({
                    "action": action,
                    "item_id": info["id"],
                } for info in self._input),
            }

        elif action in ["tags_add", "tags_replace"]:
            # FIXME: Remove me.
            @cache_today
            def __get_recent_bookmarks():
                return self._clean_get(self._parse_api_response(self._api_request({
                    "detailType": "complete",
                    "sort": "newest",
                    "count": 6000,  # get the newest 10,000 items.  # FIXME: Factor out as constant.  # FIXME: Barfs for >10K
                    "state": "all",
                }, API.RETRIEVE_URL)))
            saved_df = __get_recent_bookmarks()

            try:
                taggable_df = self._clean_get(pd.DataFrame([item.named for item in self._input]))
                trainable = ~saved_df["item_id"].isin(taggable_df["id"])

                from .topics import build_auto_tagger
                tagger = build_auto_tagger(saved_df.loc[trainable])
                tagged_df = tagger.transform(saved_df.loc[~trainable])

                tagged_df["action"] = action
                payload = {
                    "actions": tuple(tagged_df.reset_index()[["action", "item_id", "tags"]].to_dict("records")),
                }
            except:
                logging.warn("boom", exc_info=True)
                import pdb; pdb.set_trace()

            # FIXME: Remove me.
            displayable_df = pd.concat([saved_df.loc[~trainable, ["resolved_title", "excerpt", "resolved_url", "time_added"]], tagged_df], axis=1)
            logging.info("Auto-tagged %d links:\n%s", tagged_df.shape[0],
                         displayable_df.sample(50).sort_values("time_added", ascending=False))
            import pdb; pdb.set_trace()

        else:
            raise ArgumentError()

        self._payload = payload
        self._api_endpoint = API.MODIFY_URL

        self._api_request()

    def run(self):

        # validate format specifier
        self._validate_format()

        if self._args.do == "get":
            self._get()

            for info in self._output:
                self._print_to_console(info)
                if self._args.archive:
                    self._save_to_archive(info)
                elif self._args.output == "browser":
                    self._open_in_browser(info)
            else:
                if self._args.output:
                    self._output_to_file()

        else:
            if self._args.input == "console":
                self._get_console_input()
            elif self._args.input == "redirect":
                self._get_redirect_input()
            elif self._args.input.startswith("http"):
                self._get_args_input()
            else:
                self._get_file_input()

            if self._args.do == "put":
                self._put()
            elif self._args.do == "mod":
                self._modify()
