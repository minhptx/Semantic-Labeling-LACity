#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, requests, subprocess, argparse, ujson
from semantic_modeling.config import config
from terminaltables import AsciiTable

def TextMessage(text: str) -> dict:
    return {"text": text}


def ExpResultMessage(dataset: str, detail: str, exp_dir: str, average_result: dict) -> dict:
    commit_id = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode('utf-8')
    git_url = f"https://github.com/binhlvu/source-modeling/tree/{commit_id}"

    tbl = AsciiTable([
        sorted(average_result.keys()),
        ["%.3f" % average_result[k] for k in sorted(average_result.keys())]
    ])
    for i in range(len(average_result)):
        tbl.justify_columns[i] = 'center'

    return {
        "text": "new experiment result!",
        "attachments": [
            {
                "title": dataset,
                "title_link": git_url,
                "text": detail,
                "fields": [
                    {
                        "title": "Experiment directory",
                        "value": exp_dir,
                    },
                    {
                        "title": "Result",
                        "value": f"```{tbl.table}```"
                    }
                ],
                "mrkdwn_in": ["text", "pretext", "fields"]
            }
        ]
    }


def send_message(webhook_url, message):
    r = requests.post(webhook_url, json=message)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Assembling experiment')
    parser.add_argument("channel")
    parser.add_argument("message")
    parser.add_argument("params")

    args = parser.parse_args()

    if args.message == "TextMessage":
        Func = TextMessage
    else:
        Func = ExpResultMessage

    params = ujson.loads(args.params)
    send_message(config.slack.channel[args.channel], Func(**params))

    # send_message(config.slack.channel.test, ExpResultMessage("museum_edm", "assembling experiment", "/home/rook", {
    #     "precision": 0.05,
    #     "recall": 0.1,
    #     "f1": 0.5,
    #     "stype-aac": 0.78
    # }))