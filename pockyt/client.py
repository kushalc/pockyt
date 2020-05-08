from __future__ import absolute_import, print_function, unicode_literals, with_statement

import copy
import logging
import sys
import time
from os.path import join

import parse

from .api import API
from .compat import prompt
from .wrapper import Browser, FileSystem, Network

# FIXME: Remove this before submitting pull request.
try:
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

    def _api_request(self):
        # add API access credentials
        self._payload.update(self._credentials)

        # access API
        def __batch_payload(n):
            if "actions" not in self._payload:
                yield self._payload

            else:
                for ndx in range(0, len(self._payload["actions"]), n):
                    payload = copy.copy(self._payload)
                    payload["actions"] = self._payload["actions"][ndx:ndx+n]
                    yield payload

        for payload in __batch_payload(100):
            logging.debug("Executing network request: %.1000s", payload)
            self._response = Network.post_request(self._api_endpoint,
                                                  payload)

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
            self._format_spec = self._args.format + "\n"
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

        items = self._response.data.get("list", {})

        if len(items) == 0:
            print("No items found !")
            sys.exit(0)

        self._output = tuple({
            "id": item.get("item_id"),
            "title": item.get("resolved_title"),
            "link": item.get("resolved_url"),
            "excerpt": item.get("excerpt"),
            "tags": ",".join(self._process_tags(item.get("tags"))),
        } for item in items.values())

    def _process_tags(self, tags):
        if tags:
            return tags.keys()

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
            action = "tags_add"
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

        elif action == "tags_add":
            from topics import _auto_tag
            tagged = _auto_tag(self._input)

            payload = {
                "actions":
                tuple({
                    "action": action,
                    "item_id": info["id"],
                    "tags": info.named.get("tags", ""),
                } for info in tagged),
            }

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
